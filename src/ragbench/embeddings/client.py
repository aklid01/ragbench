"""Embedding client for OpenAI-compatible ``/v1/embeddings`` endpoints.

Passages and queries are embedded separately because asymmetric models want
an ``input_type`` hint; symmetric models ignore it. Batching, retries with
exponential backoff, request spacing and dimension validation all live here,
so the retrieval code only sees vectors in and vectors out.

Error handling: 408, 429 and every 5xx except 501 are retried; a ``Retry-After``
header is honored when present so the client waits exactly as long as the
server asks instead of guessing. Everything else fails immediately with an
actionable message.

Content safety screening (HTTP 422) rejects individual texts, so a rejecting
batch is bisected down to the offending inputs: everything embeddable is
embedded, the rejects are reported to the caller instead of failing the run.

The request spacing applies to every attempt, retries included, and is
enforced per client instance: concurrent processes each keep their own
budget and can exceed the configured rate together.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from ..config import EmbeddingConfig
from ..errors import EmbeddingContentError, EmbeddingError

__all__ = [
    "MAX_RETRY_AFTER_SECONDS",
    "EmbeddingBatch",
    "EmbeddingClient",
    "RejectedInput",
]

MAX_RETRY_AFTER_SECONDS = 120.0

_REJECTION_HINTS = {
    400: "the request payload was rejected (check input_type and model settings)",
    401: "authentication failed; check RAGBENCH_EMBEDDING_API_KEY",
    403: "access denied; the key may lack access to this model",
    404: "model or endpoint not found; check RAGBENCH_EMBEDDING_MODEL "
    "and RAGBENCH_EMBEDDING_BASE_URL",
    409: "conflict with the current state of the endpoint",
    413: "payload too large; lower RAGBENCH_EMBEDDING_BATCH_SIZE",
    422: "the request was unprocessable for the configured model",
    501: "the endpoint does not implement this operation",
}


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class RejectedInput:
    """One text the endpoint refused to embed."""

    index: int
    text: str
    message: str


@dataclass
class EmbeddingBatch:
    """Vectors for the accepted inputs and the reasons for the rejected ones."""

    vectors: list[list[float]] = field(default_factory=list)
    rejected: list[RejectedInput] = field(default_factory=list)


def _is_retryable(status: int) -> bool:
    return status in {408, 429} or (500 <= status <= 599 and status != 501)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse the ``Retry-After`` header, in seconds or HTTP-date form."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return float(raw)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (when - _now_utc()).total_seconds()


class EmbeddingClient:
    """Embed passages and queries through one configured endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._monotonic = monotonic
        rpm = config.requests_per_minute
        self._min_request_interval = 60.0 / rpm if rpm else 0.0
        self._last_request_at: float | None = None
        headers = {}
        if config.api_key is not None:
            headers["Authorization"] = f"Bearer {config.api_key.get_secret_value()}"
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            transport=transport,
            headers=headers,
            follow_redirects=True,
        )

    @property
    def model(self) -> str:
        return self.config.model

    def close(self) -> None:
        self._client.close()

    def embed_passages(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed document chunks, in batches of ``batch_size``.

        Inputs rejected by content safety screening are isolated by bisection
        and reported in ``rejected``; the rest are embedded.
        """
        if not texts:
            return EmbeddingBatch()
        self._require_key()
        batch = EmbeddingBatch()
        for start in range(0, len(texts), self.config.batch_size):
            chunk_texts = list(texts[start : start + self.config.batch_size])
            self._embed_tolerant(chunk_texts, start, self.config.passage_input_type, batch)
        return batch

    def embed_query(self, text: str) -> list[float]:
        """Embed one query. A rejected query is fatal: there is nothing to retrieve with."""
        self._require_key()
        return self._request([text], self.config.query_input_type)[0]

    def _embed_tolerant(
        self,
        texts: list[str],
        offset: int,
        input_type: str | None,
        batch: EmbeddingBatch,
    ) -> None:
        try:
            batch.vectors.extend(self._request(texts, input_type))
            return
        except EmbeddingContentError as exc:
            if len(texts) == 1:
                batch.rejected.append(RejectedInput(offset, texts[0], str(exc)))
                return
            middle = len(texts) // 2
            self._embed_tolerant(texts[:middle], offset, input_type, batch)
            self._embed_tolerant(texts[middle:], offset + middle, input_type, batch)

    def _require_key(self) -> None:
        if self.config.api_key is None:
            raise EmbeddingError(
                "embedding api key is not set (RAGBENCH_EMBEDDING_API_KEY); "
                f"base_url is {self.config.base_url}"
            )

    def _throttle(self) -> None:
        if self._min_request_interval <= 0.0:
            return
        now = self._monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            if elapsed < self._min_request_interval:
                self._sleep(self._min_request_interval - elapsed)
        self._last_request_at = self._monotonic()

    def _request(self, batch: list[str], input_type: str | None) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.config.model, "input": batch}
        if input_type:
            payload["input_type"] = input_type

        last_error = "no attempt was made"
        for attempt in range(self.config.max_retries + 1):
            self._throttle()
            try:
                response = self._client.post("/embeddings", json=payload)
            except httpx.TimeoutException:
                last_error = (
                    f"timed out after {self.config.timeout_seconds:g}s; if this endpoint "
                    f"is cold starting, raise RAGBENCH_EMBEDDING_TIMEOUT_SECONDS"
                )
            except httpx.HTTPError as exc:
                last_error = f"request failed: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, len(batch))
                if not _is_retryable(response.status_code):
                    body = response.text
                    lowered = body.lower()
                    if response.status_code == 422 and (
                        "content safety" in lowered or "screening" in lowered
                    ):
                        raise EmbeddingContentError(
                            f"input rejected by content safety screening (HTTP 422): {body[:200]}"
                        )
                    raise EmbeddingError(self._rejection_message(response.status_code, body))
                last_error = f"HTTP {response.status_code} from the embedding endpoint"
                if attempt < self.config.max_retries:
                    self._sleep(self._retry_delay(response, attempt))
                continue
            if attempt < self.config.max_retries:
                self._sleep(self.config.retry_base_delay_seconds * (2**attempt))
        raise EmbeddingError(
            f"embedding endpoint failed after {self.config.max_retries + 1} attempts: {last_error}"
        )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Wait exactly as long as the server asks, or back off exponentially."""
        retry_after = _retry_after_seconds(response)
        if retry_after is None:
            return self.config.retry_base_delay_seconds * (2**attempt)
        if retry_after > MAX_RETRY_AFTER_SECONDS:
            raise EmbeddingError(
                f"the endpoint asked to retry after {retry_after:.0f}s, which exceeds the "
                f"{MAX_RETRY_AFTER_SECONDS:g}s cap; lower RAGBENCH_REQUESTS_PER_MINUTE or "
                f"try again later"
            )
        return max(retry_after, 0.0)

    @staticmethod
    def _rejection_message(status: int, body: str) -> str:
        hint = _REJECTION_HINTS.get(status, "the endpoint rejected the request")
        return f"embedding endpoint returned HTTP {status} ({hint}): {body[:200]}"

    def _parse(self, response: httpx.Response, expected_count: int) -> list[list[float]]:
        try:
            items = response.json()["data"]
            ordered = sorted(items, key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise EmbeddingError(f"embedding response is not understood: {exc}") from exc
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"embedding response holds {len(vectors)} vectors for {expected_count} inputs"
            )
        expected = self.config.expected_dimensions
        if expected is not None:
            actual = {len(vector) for vector in vectors}
            if actual != {expected}:
                raise EmbeddingError(
                    f"embedding dimensions {sorted(actual)} do not match the configured {expected}"
                )
        return vectors

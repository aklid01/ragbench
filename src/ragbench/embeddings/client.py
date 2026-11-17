"""Embedding client for OpenAI-compatible ``/v1/embeddings`` endpoints.

Passages and queries are embedded separately because asymmetric models want
an ``input_type`` hint; symmetric models ignore it. Batching, request spacing
and dimension validation all live here, so the retrieval code only sees
vectors in and vectors out; the retry semantics are the shared endpoint
client's.

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

import httpx

from ..config import EmbeddingConfig
from ..endpoint_client import EndpointClient
from ..errors import EmbeddingContentError, EmbeddingError

__all__ = [
    "EmbeddingBatch",
    "EmbeddingClient",
    "RejectedInput",
]

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


class EmbeddingClient(EndpointClient[list[list[float]], EmbeddingConfig]):
    """Embed passages and queries through one configured endpoint."""

    key_label = "embedding"
    key_env_var = "RAGBENCH_EMBEDDING_API_KEY"
    timeout_env_var = "RAGBENCH_EMBEDDING_TIMEOUT_SECONDS"
    endpoint_label = "embedding endpoint"
    rejection_hints = _REJECTION_HINTS
    retry_cap_hint = "lower RAGBENCH_REQUESTS_PER_MINUTE or try again later"
    error_factory = EmbeddingError

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config, transport=transport, sleep=sleep, monotonic=monotonic)
        rpm = config.requests_per_minute
        self._min_request_interval = 60.0 / rpm if rpm else 0.0
        self._last_request_at: float | None = None
        self._expected_count = 0

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

    def _before_attempt(self) -> None:
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
        self._expected_count = len(batch)
        return self._post_with_retries("/embeddings", payload)

    def _handle_success(self, response: httpx.Response) -> list[list[float]]:
        return self._parse(response, self._expected_count)

    def _fail_non_retryable(self, response: httpx.Response) -> None:
        body = response.text
        lowered = body.lower()
        if response.status_code == 422 and ("content safety" in lowered or "screening" in lowered):
            raise EmbeddingContentError(
                f"input rejected by content safety screening (HTTP 422): {body[:200]}"
            )
        raise EmbeddingError(self._rejection_message(response.status_code, body))

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

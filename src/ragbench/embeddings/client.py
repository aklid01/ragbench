"""Embedding client for OpenAI-compatible ``/v1/embeddings`` endpoints.

Passages and queries are embedded separately because asymmetric models want
an ``input_type`` hint; symmetric models ignore it. Batching, retries with
exponential backoff, request spacing and dimension validation all live here,
so the retrieval code only sees vectors in and vectors out.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import httpx

from ..config import EmbeddingConfig
from ..errors import EmbeddingError

__all__ = ["RETRYABLE_STATUS_CODES", "EmbeddingClient"]

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


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
        )

    @property
    def model(self) -> str:
        return self.config.model

    def close(self) -> None:
        self._client.close()

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document chunks, in batches of ``batch_size``."""
        return self._embed(texts, self.config.passage_input_type)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return self._embed([text], self.config.query_input_type)[0]

    def _embed(self, texts: Sequence[str], input_type: str | None) -> list[list[float]]:
        if not texts:
            return []
        self._require_key()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            vectors.extend(self._request(batch, input_type))
        return vectors

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
        self._throttle()
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post("/embeddings", json=payload)
            except httpx.HTTPError as exc:
                last_error = f"request failed: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, len(batch))
                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code} from the embedding endpoint"
                else:
                    raise EmbeddingError(
                        f"embedding endpoint rejected the request "
                        f"(HTTP {response.status_code}): {response.text[:200]}"
                    )
            if attempt < self.config.max_retries:
                self._sleep(self.config.retry_base_delay_seconds * (2**attempt))
        raise EmbeddingError(
            f"embedding endpoint failed after {self.config.max_retries + 1} attempts: {last_error}"
        )

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

"""LLM client for OpenAI-compatible ``/v1/chat/completions`` endpoints.

The retry semantics mirror the embedding client: 408, 429 and every 5xx
except 501 are retried, a ``Retry-After`` header is honored when present, and
everything else fails immediately with an actionable message. Latency, the
reported model and token usage are captured on every successful call so the
evaluation layer can account for them without re-parsing responses.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from ..config import LLMConfig
from ..errors import GenerationError

__all__ = [
    "MAX_RETRY_AFTER_SECONDS",
    "LLMCompletion",
    "LLMClient",
]

MAX_RETRY_AFTER_SECONDS = 120.0

_REJECTION_HINTS = {
    400: "the request payload was rejected (check the model name and message format)",
    401: "authentication failed; check RAGBENCH_LLM_API_KEY",
    403: "access denied; the key may lack access to this model",
    404: "model or endpoint not found; check RAGBENCH_LLM_MODEL and RAGBENCH_LLM_BASE_URL",
    409: "conflict with the current state of the endpoint",
    413: "payload too large; lower RAGBENCH_LLM_MAX_OUTPUT_TOKENS or shorten the prompt",
    422: "the request was unprocessable for the configured model",
    501: "the endpoint does not implement this operation",
}


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class LLMCompletion:
    """One successful chat completion with its accounting."""

    content: str
    finish_reason: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float


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


class LLMClient:
    """Complete chat messages through one configured endpoint."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._monotonic = monotonic
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

    def complete(self, messages: Sequence[Mapping[str, str]]) -> LLMCompletion:
        """Send the messages and return the first choice with its accounting."""
        if not messages:
            raise GenerationError("a completion needs at least one message")
        self._require_key()
        payload = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }

        last_error = "no attempt was made"
        for attempt in range(self.config.max_retries + 1):
            started = self._monotonic()
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                last_error = (
                    f"timed out after {self.config.timeout_seconds:g}s; if this endpoint "
                    f"is cold starting, raise RAGBENCH_LLM_TIMEOUT_SECONDS"
                )
            except httpx.HTTPError as exc:
                last_error = f"request failed: {exc}"
            else:
                if response.status_code == 200:
                    return self._parse(response, self._monotonic() - started)
                if not _is_retryable(response.status_code):
                    raise GenerationError(
                        self._rejection_message(response.status_code, response.text)
                    )
                last_error = f"HTTP {response.status_code} from the LLM endpoint"
                if attempt < self.config.max_retries:
                    self._sleep(self._retry_delay(response, attempt))
                continue
            if attempt < self.config.max_retries:
                self._sleep(self.config.retry_base_delay_seconds * (2**attempt))
        raise GenerationError(
            f"LLM endpoint failed after {self.config.max_retries + 1} attempts: {last_error}"
        )

    def _require_key(self) -> None:
        if self.config.api_key is None:
            raise GenerationError(
                f"LLM api key is not set (RAGBENCH_LLM_API_KEY); base_url is {self.config.base_url}"
            )

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Wait exactly as long as the server asks, or back off exponentially."""
        retry_after = _retry_after_seconds(response)
        if retry_after is None:
            return self.config.retry_base_delay_seconds * (2**attempt)
        if retry_after > MAX_RETRY_AFTER_SECONDS:
            raise GenerationError(
                f"the endpoint asked to retry after {retry_after:.0f}s, which exceeds the "
                f"{MAX_RETRY_AFTER_SECONDS:g}s cap; try again later"
            )
        return max(retry_after, 0.0)

    @staticmethod
    def _rejection_message(status: int, body: str) -> str:
        hint = _REJECTION_HINTS.get(status, "the endpoint rejected the request")
        return f"LLM endpoint returned HTTP {status} ({hint}): {body[:200]}"

    def _parse(self, response: httpx.Response, latency_seconds: float) -> LLMCompletion:
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            content = message["content"]
            finish_reason = choice.get("finish_reason")
            reported_model = body.get("model")
            usage = body.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise GenerationError(f"LLM response is not understood: {exc}") from exc
        if not isinstance(content, str):
            raise GenerationError("LLM response holds no message content")
        return LLMCompletion(
            content=content,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            model=reported_model if isinstance(reported_model, str) else None,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
            latency_seconds=latency_seconds,
        )

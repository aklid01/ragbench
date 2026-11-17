"""LLM client for OpenAI-compatible ``/v1/chat/completions`` endpoints.

The retry semantics are the shared endpoint client's. Latency, the reported
model and token usage are captured on every successful call so the
evaluation layer can account for them without re-parsing responses.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import httpx

from ..config import LLMConfig
from ..endpoint_client import EndpointClient
from ..errors import GenerationError

__all__ = [
    "LLMCompletion",
    "LLMClient",
]

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


class LLMClient(EndpointClient[LLMCompletion, LLMConfig]):
    """Complete chat messages through one configured endpoint."""

    key_label = "LLM"
    key_env_var = "RAGBENCH_LLM_API_KEY"
    timeout_env_var = "RAGBENCH_LLM_TIMEOUT_SECONDS"
    endpoint_label = "LLM endpoint"
    rejection_hints = _REJECTION_HINTS
    error_factory = GenerationError

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(config, transport=transport, sleep=sleep, monotonic=monotonic)
        self._started = 0.0

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
        return self._post_with_retries("/chat/completions", payload)

    def _before_attempt(self) -> None:
        self._started = self._monotonic()

    def _handle_success(self, response: httpx.Response) -> LLMCompletion:
        return self._parse(response, self._monotonic() - self._started)

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

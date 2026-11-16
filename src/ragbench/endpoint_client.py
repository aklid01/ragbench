"""Shared scaffolding for the OpenAI-compatible HTTP endpoint clients.

Retry semantics live here once: 408, 429 and every 5xx except 501 are
retried, a ``Retry-After`` header is honored when present so the client
waits exactly as long as the server asks, and everything else fails
immediately with an actionable message. The embedding and generation
clients contribute the payload shape, the response parsing and the
endpoint specific hints.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import ClassVar

import httpx

from .config import _BaseProviderConfig

__all__ = [
    "MAX_RETRY_AFTER_SECONDS",
    "EndpointClient",
    "is_retryable",
    "now_utc",
    "retry_after_seconds",
]

MAX_RETRY_AFTER_SECONDS = 120.0


def now_utc() -> datetime:
    return datetime.now(UTC)


def is_retryable(status: int) -> bool:
    return status in {408, 429} or (500 <= status <= 599 and status != 501)


def retry_after_seconds(response: httpx.Response) -> float | None:
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
    return (when - now_utc()).total_seconds()


class EndpointClient[T]:
    """One configured endpoint with auth headers, request retries and backoff.

    Subclasses set the class level hints and implement
    :meth:`_handle_success`; :meth:`_post_with_retries` owns the retry loop.
    """

    key_label: ClassVar[str]
    key_env_var: ClassVar[str]
    timeout_env_var: ClassVar[str]
    endpoint_label: ClassVar[str]
    rejection_hints: ClassVar[dict[int, str]] = {}
    retry_cap_hint: ClassVar[str] = "try again later"
    error_factory: ClassVar[type[Exception]]

    def __init__(
        self,
        config: _BaseProviderConfig,
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

    def _require_key(self) -> None:
        if self.config.api_key is None:
            raise self.error_factory(
                f"{self.key_label} api key is not set ({self.key_env_var}); "
                f"base_url is {self.config.base_url}"
            )

    def _post_with_retries(self, path: str, payload: dict[str, object]) -> T:
        """POST ``payload``, retrying what is retryable, and parse the answer."""
        last_error = "no attempt was made"
        for attempt in range(self.config.max_retries + 1):
            self._before_attempt()
            try:
                response = self._client.post(path, json=payload)
            except httpx.TimeoutException:
                last_error = (
                    f"timed out after {self.config.timeout_seconds:g}s; if this endpoint "
                    f"is cold starting, raise {self.timeout_env_var}"
                )
            except httpx.HTTPError as exc:
                last_error = f"request failed: {exc}"
            else:
                if response.status_code == 200:
                    return self._handle_success(response)
                if not is_retryable(response.status_code):
                    self._fail_non_retryable(response)
                last_error = f"HTTP {response.status_code} from the {self.endpoint_label}"
                if attempt < self.config.max_retries:
                    self._sleep(self._retry_delay(response, attempt))
                continue
            if attempt < self.config.max_retries:
                self._sleep(self.config.retry_base_delay_seconds * (2**attempt))
        raise self.error_factory(
            f"{self.endpoint_label} failed after "
            f"{self.config.max_retries + 1} attempts: {last_error}"
        )

    def _before_attempt(self) -> None:
        """Hook run before every request, retries included."""

    def _handle_success(self, response: httpx.Response) -> T:
        raise NotImplementedError

    def _fail_non_retryable(self, response: httpx.Response) -> None:
        raise self.error_factory(self._rejection_message(response.status_code, response.text))

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        """Wait exactly as long as the server asks, or back off exponentially."""
        retry_after = retry_after_seconds(response)
        if retry_after is None:
            return self.config.retry_base_delay_seconds * (2**attempt)
        if retry_after > MAX_RETRY_AFTER_SECONDS:
            raise self.error_factory(
                f"the endpoint asked to retry after {retry_after:.0f}s, which exceeds the "
                f"{MAX_RETRY_AFTER_SECONDS:g}s cap; {self.retry_cap_hint}"
            )
        return max(retry_after, 0.0)

    def _rejection_message(self, status: int, body: str) -> str:
        hint = self.rejection_hints.get(status, "the endpoint rejected the request")
        return f"{self.endpoint_label} returned HTTP {status} ({hint}): {body[:200]}"

"""LLM client: retries, error classification, response parsing and accounting."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from ragbench.config import LLMConfig
from ragbench.errors import GenerationError
from ragbench.generation import LLMClient


def make_config(**overrides: object) -> LLMConfig:
    values: dict[str, object] = {
        "base_url": "http://llm.test/v1",
        "model": "test-llm",
        "api_key": SecretStr("k"),
        "max_retries": 2,
        "retry_base_delay_seconds": 1.0,
    }
    values.update(overrides)
    return LLMConfig(**values)  # type: ignore[arg-type]


def completion_body(
    content: str = "an answer [1]",
    model: str = "reported-llm",
    usage: dict[str, int] | None = None,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    }


def ok_transport(handler: object) -> httpx.MockTransport:
    return httpx.MockTransport(handler)  # type: ignore[arg-type]


def messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "answer from the passages"},
        {"role": "user", "content": "question with passages"},
    ]


def test_completion_is_parsed_with_accounting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body(), request=request)

    client = LLMClient(make_config(), transport=ok_transport(handler))
    result = client.complete(messages())
    assert result.content == "an answer [1]"
    assert result.finish_reason == "stop"
    assert result.model == "reported-llm"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (120, 45, 165)
    assert result.latency_seconds >= 0.0
    client.close()


def test_missing_usage_is_reported_as_none() -> None:
    body = completion_body()
    body["usage"] = None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body, request=request)

    client = LLMClient(make_config(), transport=ok_transport(handler))
    result = client.complete(messages())
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
    client.close()


def test_payload_carries_messages_temperature_and_token_cap() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=completion_body(), request=request)

    client = LLMClient(
        make_config(temperature=0.4, max_output_tokens=256), transport=ok_transport(handler)
    )
    client.complete(messages())
    assert seen["model"] == "test-llm"
    assert seen["temperature"] == 0.4
    assert seen["max_tokens"] == 256
    assert seen["messages"] == messages()
    client.close()


def test_empty_messages_fail_without_a_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    client = LLMClient(make_config(), transport=ok_transport(handler))
    with pytest.raises(GenerationError) as excinfo:
        client.complete([])
    assert "at least one message" in str(excinfo.value)
    client.close()


def test_missing_api_key_fails_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    client = LLMClient(make_config(api_key=None), transport=ok_transport(handler))
    with pytest.raises(GenerationError) as excinfo:
        client.complete(messages())
    assert "api key" in str(excinfo.value)
    assert "RAGBENCH_LLM_API_KEY" in str(excinfo.value)
    client.close()


def test_retryable_status_is_retried_with_backoff_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            return httpx.Response(429, json={"error": "slow down"}, request=request)
        return httpx.Response(200, json=completion_body(), request=request)

    sleeps: list[float] = []
    client = LLMClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    result = client.complete(messages())
    assert result.content == "an answer [1]"
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]
    client.close()


def test_retry_exhaustion_raises_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"}, request=request)

    sleeps: list[float] = []
    client = LLMClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    with pytest.raises(GenerationError) as excinfo:
        client.complete(messages())
    assert "3 attempts" in str(excinfo.value)
    assert "503" in str(excinfo.value)
    assert sleeps == [1.0, 2.0]
    client.close()


def test_client_errors_fail_without_retrying() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(401, json={"error": "bad key"}, request=request)

    client = LLMClient(make_config(), transport=ok_transport(handler))
    with pytest.raises(GenerationError) as excinfo:
        client.complete(messages())
    assert attempts["count"] == 1
    assert "401" in str(excinfo.value)
    assert "RAGBENCH_LLM_API_KEY" in str(excinfo.value)
    client.close()


def test_timeout_is_reported_with_the_config_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    client = LLMClient(
        make_config(max_retries=0, timeout_seconds=5.0), transport=ok_transport(handler)
    )
    with pytest.raises(GenerationError) as excinfo:
        client.complete(messages())
    assert "timed out after 5s" in str(excinfo.value)
    assert "RAGBENCH_LLM_TIMEOUT_SECONDS" in str(excinfo.value)
    client.close()


def test_connection_failure_is_reported_after_the_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = LLMClient(make_config(), transport=ok_transport(handler), sleep=lambda _: None)
    with pytest.raises(GenerationError) as excinfo:
        client.complete(messages())
    assert "3 attempts" in str(excinfo.value)
    assert "refused" in str(excinfo.value)
    client.close()


def retry_after_handler(status: int, headers: dict[str, str], succeed_on: int | None = None):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if succeed_on is not None and attempts["count"] >= succeed_on:
            return httpx.Response(200, json=completion_body(), request=request)
        return httpx.Response(status, headers=headers, request=request)

    return attempts, handler


def test_429_honors_retry_after_seconds() -> None:
    attempts, handler = retry_after_handler(429, {"Retry-After": "7"}, succeed_on=3)
    sleeps: list[float] = []
    client = LLMClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    client.complete(messages())
    assert attempts["count"] == 3
    assert sleeps == [7.0, 7.0]
    client.close()


def test_retry_after_over_the_cap_fails_fast() -> None:
    attempts, handler = retry_after_handler(429, {"Retry-After": "3600"})
    sleeps: list[float] = []
    client = LLMClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    with pytest.raises(GenerationError) as excinfo:
        client.complete(messages())
    assert "cap" in str(excinfo.value)
    assert attempts["count"] == 1
    assert sleeps == []
    client.close()


def test_malformed_retry_after_falls_back_to_backoff() -> None:
    attempts, handler = retry_after_handler(429, {"Retry-After": "garbage"}, succeed_on=2)
    sleeps: list[float] = []
    client = LLMClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    client.complete(messages())
    assert sleeps == [1.0]
    client.close()


def test_permanent_client_errors_fail_fast_with_hints() -> None:
    expected = {
        400: "payload was rejected",
        401: "RAGBENCH_LLM_API_KEY",
        403: "access denied",
        404: "RAGBENCH_LLM_MODEL",
        409: "conflict",
        413: "RAGBENCH_LLM_MAX_OUTPUT_TOKENS",
        422: "unprocessable",
        501: "does not implement",
    }
    for status, hint in expected.items():
        attempts, handler = retry_after_handler(status, {})
        client = LLMClient(make_config(), transport=ok_transport(handler))
        with pytest.raises(GenerationError) as excinfo:
            client.complete(messages())
        assert f"HTTP {status}" in str(excinfo.value)
        assert hint in str(excinfo.value)
        assert attempts["count"] == 1
        client.close()


def test_transient_server_errors_are_retried() -> None:
    for status in (408, 500, 502, 503, 504):
        attempts, handler = retry_after_handler(status, {}, succeed_on=3)
        client = LLMClient(make_config(), transport=ok_transport(handler))
        client.complete(messages())
        assert attempts["count"] == 3
        client.close()


def test_malformed_response_body_is_rejected() -> None:
    for body in ({}, {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": 42}}]}):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body, request=request)

        client = LLMClient(make_config(), transport=ok_transport(handler))
        with pytest.raises(GenerationError) as excinfo:
            client.complete(messages())
        message = str(excinfo.value)
        assert "not understood" in message or "no message content" in message
        client.close()

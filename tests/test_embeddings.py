"""Embedding client: batching, retries, rate limiting and response validation."""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from ragbench.config import EmbeddingConfig
from ragbench.embeddings import EmbeddingClient
from ragbench.errors import EmbeddingError


def make_config(**overrides: object) -> EmbeddingConfig:
    values: dict[str, object] = {
        "base_url": "http://embedding.test/v1",
        "model": "test-embed",
        "api_key": SecretStr("k"),
        "batch_size": 2,
        "max_retries": 2,
        "retry_base_delay_seconds": 1.0,
        "expected_dimensions": 4,
        "requests_per_minute": None,
    }
    values.update(overrides)
    return EmbeddingConfig(**values)  # type: ignore[arg-type]


def ok_transport(handler: object) -> httpx.MockTransport:
    return httpx.MockTransport(handler)  # type: ignore[arg-type]


def embedding_handler(vectors: list[list[float]]):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
                ]
            },
            request=request,
        )
        return payload

    return handler


def test_batches_are_split_and_order_preserved() -> None:
    seen_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = request.read()
        import json

        seen_sizes.append(len(json.loads(inputs)["input"]))
        vectors = [[1.0, 0.0, 0.0, 0.0]] * len(json.loads(inputs)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]},
            request=request,
        )

    client = EmbeddingClient(make_config(), transport=ok_transport(handler))
    vectors = client.embed_passages(["a", "b", "c", "d", "e"]).vectors
    assert seen_sizes == [2, 2, 1]
    assert len(vectors) == 5
    assert vectors[4] == [1.0, 0.0, 0.0, 0.0]
    client.close()


def test_retryable_status_is_retried_with_backoff_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            return httpx.Response(429, json={"error": "slow down"}, request=request)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.5, 0.5, 0.5, 0.5]}]},
            request=request,
        )

    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    vectors = client.embed_query("hello")
    assert vectors == [0.5, 0.5, 0.5, 0.5]
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]
    client.close()


def test_retry_exhaustion_raises_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"}, request=request)

    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_query("hello")
    assert "3 attempts" in str(excinfo.value)
    assert "503" in str(excinfo.value)
    assert sleeps == [1.0, 2.0]
    client.close()


def test_client_errors_fail_without_retrying() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(401, json={"error": "bad key"}, request=request)

    client = EmbeddingClient(make_config(), transport=ok_transport(handler))
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_query("hello")
    assert attempts["count"] == 1
    assert "401" in str(excinfo.value)
    client.close()


def test_dimension_mismatch_is_rejected() -> None:
    client = EmbeddingClient(
        make_config(), transport=ok_transport(embedding_handler([[1.0, 0.0, 0.0]]))
    )
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_query("hello")
    assert "dimensions" in str(excinfo.value)
    client.close()


def test_wrong_vector_count_is_rejected() -> None:
    client = EmbeddingClient(
        make_config(batch_size=2),
        transport=ok_transport(embedding_handler([[1.0, 0.0, 0.0, 0.0]])),
    )
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_passages(["a", "b"])
    assert "1 vectors for 2 inputs" in str(excinfo.value)
    client.close()


def test_empty_batch_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    client = EmbeddingClient(make_config(), transport=ok_transport(handler))
    assert client.embed_passages([]).vectors == []
    assert client.embed_passages([]).rejected == []
    client.close()


def test_missing_api_key_fails_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    client = EmbeddingClient(make_config(api_key=None), transport=ok_transport(handler))
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_query("hello")
    assert "api key" in str(excinfo.value)
    client.close()


def test_input_type_is_sent_only_when_configured() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read())
        seen.append(body.get("input_type"))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]},
            request=request,
        )

    client = EmbeddingClient(
        make_config(query_input_type="query", passage_input_type="passage"),
        transport=ok_transport(handler),
    )
    client.embed_query("what?")
    client.embed_passages(["passage text"])
    assert seen == ["query", "passage"]
    client.close()


def test_input_type_is_omitted_when_not_configured() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read())
        seen.append(body.get("input_type"))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]},
            request=request,
        )

    client = EmbeddingClient(make_config(), transport=ok_transport(handler))
    client.embed_query("what?")
    assert seen == [None]
    client.close()


def test_requests_are_spaced_according_to_the_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]})

    clock = iter([0.0, 0.0, 0.1, 0.1])
    sleeps: list[float] = []
    client = EmbeddingClient(
        make_config(requests_per_minute=120.0, batch_size=1),
        transport=ok_transport(handler),
        sleep=sleeps.append,
        monotonic=lambda: next(clock),
    )
    client.embed_passages(["a", "b"])
    assert sleeps == [0.4]
    client.close()


def test_retries_are_spaced_by_the_rate_limit_too() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, json={"error": "slow down"}, request=request)
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]}, request=request
        )

    clock = iter([0.0, 0.0, 0.1, 0.1])
    sleeps: list[float] = []
    client = EmbeddingClient(
        make_config(requests_per_minute=120.0, retry_base_delay_seconds=5.0),
        transport=ok_transport(handler),
        sleep=sleeps.append,
        monotonic=lambda: next(clock),
    )
    client.embed_query("hello")
    assert attempts["count"] == 2
    assert sleeps == [5.0, 0.4]
    client.close()


def retry_after_handler(status: int, headers: dict[str, str], succeed_on: int | None = None):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if succeed_on is not None and attempts["count"] >= succeed_on:
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]}
            )
        return httpx.Response(status, headers=headers, request=request)

    return attempts, handler


def test_429_honors_retry_after_seconds() -> None:
    attempts, handler = retry_after_handler(429, {"Retry-After": "7"}, succeed_on=3)
    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    vectors = client.embed_query("hello")
    assert vectors == [1.0, 0.0, 0.0, 0.0]
    assert attempts["count"] == 3
    assert sleeps == [7.0, 7.0]
    client.close()


def test_retry_after_http_date_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    import ragbench.endpoint_client as endpoint_module

    frozen = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(endpoint_module, "now_utc", lambda: frozen)
    retry_at = format_datetime(frozen + timedelta(seconds=9))

    attempts, handler = retry_after_handler(503, {"Retry-After": retry_at}, succeed_on=2)
    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    client.embed_query("hello")
    assert sleeps == [9.0]
    client.close()


def test_retry_after_over_the_cap_fails_fast() -> None:
    attempts, handler = retry_after_handler(429, {"Retry-After": "3600"})
    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_query("hello")
    assert "cap" in str(excinfo.value)
    assert attempts["count"] == 1
    assert sleeps == []
    client.close()


def test_malformed_retry_after_falls_back_to_backoff() -> None:
    attempts, handler = retry_after_handler(429, {"Retry-After": "garbage"}, succeed_on=2)
    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    client.embed_query("hello")
    assert sleeps == [1.0]
    client.close()


def test_past_retry_after_date_waits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    import ragbench.endpoint_client as endpoint_module

    frozen = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    past_date = format_datetime(frozen - timedelta(seconds=30))
    monkeypatch.setattr(endpoint_module, "now_utc", lambda: frozen)

    attempts, handler = retry_after_handler(429, {"Retry-After": past_date}, succeed_on=2)
    sleeps: list[float] = []
    client = EmbeddingClient(make_config(), transport=ok_transport(handler), sleep=sleeps.append)
    client.embed_query("hello")
    assert sleeps == [0.0]
    client.close()


def test_permanent_client_errors_fail_fast_with_hints() -> None:
    expected = {
        400: "payload was rejected",
        401: "RAGBENCH_EMBEDDING_API_KEY",
        403: "access denied",
        404: "RAGBENCH_EMBEDDING_MODEL",
        409: "conflict",
        413: "RAGBENCH_EMBEDDING_BATCH_SIZE",
        422: "unprocessable",
        501: "does not implement",
    }
    for status, hint in expected.items():
        attempts, handler = retry_after_handler(status, {})
        client = EmbeddingClient(make_config(), transport=ok_transport(handler))
        with pytest.raises(EmbeddingError) as excinfo:
            client.embed_query("hello")
        assert f"HTTP {status}" in str(excinfo.value)
        assert hint in str(excinfo.value)
        assert attempts["count"] == 1
        client.close()


def test_transient_server_errors_are_retried() -> None:
    for status in (408, 500, 502, 503, 504):
        attempts, handler = retry_after_handler(status, {}, succeed_on=3)
        client = EmbeddingClient(make_config(), transport=ok_transport(handler))
        client.embed_query("hello")
        assert attempts["count"] == 3
        client.close()


def screening_handler(reject_texts: set[str]):
    """Return 422 content-safety for the marked texts, 200 otherwise."""
    requests = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        requests["count"] += 1
        inputs = json.loads(request.read())["input"]
        if any(text in reject_texts for text in inputs):
            return httpx.Response(
                422,
                json={"error": {"message": "input rejected by content safety screening"}},
            )
        vectors = [[1.0, 0.0, 0.0, 0.0]] * len(inputs)
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]},
        )

    return requests, handler


def test_one_rejection_in_a_batch_is_isolated_by_bisection() -> None:
    texts = ["fine one", "bad apple", "fine three", "fine four"]
    requests, handler = screening_handler({"bad apple"})
    client = EmbeddingClient(make_config(batch_size=4), transport=ok_transport(handler))
    result = client.embed_passages(texts)
    assert len(result.vectors) == 3
    assert len(result.rejected) == 1
    assert result.rejected[0].index == 1
    assert result.rejected[0].text == "bad apple"
    assert "content safety" in result.rejected[0].message
    assert requests["count"] == 5
    client.close()


def test_several_rejections_in_one_batch_are_all_isolated() -> None:
    texts = ["bad one", "fine", "bad two", "fine too", "bad three"]
    requests, handler = screening_handler({"bad one", "bad two", "bad three"})
    client = EmbeddingClient(make_config(batch_size=8), transport=ok_transport(handler))
    result = client.embed_passages(texts)
    assert [item.index for item in result.rejected] == [0, 2, 4]
    assert {item.text for item in result.rejected} == {"bad one", "bad two", "bad three"}
    assert len(result.vectors) == 2
    client.close()


def test_schema_422_fails_instead_of_isolating() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": {"message": "input_type must be one of 'query', 'passage'"}},
        )

    client = EmbeddingClient(make_config(), transport=ok_transport(handler))
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_passages(["a", "b"])
    assert "content safety" not in str(excinfo.value)
    assert "unprocessable" in str(excinfo.value)
    client.close()


def test_rejected_query_is_fatal() -> None:
    requests, handler = screening_handler({"bad question"})
    client = EmbeddingClient(make_config(), transport=ok_transport(handler))
    with pytest.raises(EmbeddingError) as excinfo:
        client.embed_query("bad question")
    assert "content safety" in str(excinfo.value)
    client.close()

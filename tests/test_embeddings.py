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
                    {"index": index, "embedding": vector}
                    for index, vector in enumerate(vectors)
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
    vectors = client.embed_passages(["a", "b", "c", "d", "e"])
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
    client = EmbeddingClient(
        make_config(), transport=ok_transport(handler), sleep=sleeps.append
    )
    vectors = client.embed_query("hello")
    assert vectors == [0.5, 0.5, 0.5, 0.5]
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]
    client.close()


def test_retry_exhaustion_raises_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"}, request=request)

    sleeps: list[float] = []
    client = EmbeddingClient(
        make_config(), transport=ok_transport(handler), sleep=sleeps.append
    )
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
    assert client.embed_passages([]) == []
    client.close()


def test_missing_api_key_fails_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    client = EmbeddingClient(
        make_config(api_key=None), transport=ok_transport(handler)
    )
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
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]}]}
        )

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

"""The answer assembly: completions become abstentions, citations or flags."""

from __future__ import annotations

from conftest import make_chunk
from ragbench.generation import LLMCompletion, build_answer


def make_completion(content: str, **overrides: object) -> LLMCompletion:
    values: dict[str, object] = {
        "content": content,
        "finish_reason": "stop",
        "model": "reported-llm",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "latency_seconds": 0.5,
    }
    values.update(overrides)
    return LLMCompletion(**values)  # type: ignore[arg-type]


RETRIEVED = [
    make_chunk(1, "report.pdf", 1, "The company targets net zero emissions by 2040."),
    make_chunk(2, "report.pdf", 2, "Water consumption fell by 18 percent in 2021."),
]


def test_a_cited_completion_assembles_into_a_structured_answer() -> None:
    completion = make_completion("The target is 2040 [1]. Water use fell 18 percent [2].")
    result = build_answer(
        "what is the target?", "custom-800-80", 5, RETRIEVED, completion, 0.25, "test-embed"
    )

    assert result.abstained is False
    assert result.answer == completion.content
    assert [citation.reference for citation in result.citations] == [1, 2]
    assert result.invalid_references == []
    assert len(result.retrieved) == 2
    assert result.spec == "custom-800-80"
    assert result.top_k == 5
    assert result.embedding_model == "test-embed"
    assert result.model == "reported-llm"
    assert result.prompt_tokens == 100
    assert result.total_tokens == 120
    assert result.retrieval_seconds == 0.25
    assert result.generation_seconds == 0.5
    assert result.answered_at is not None
    assert result.finish_reason == "stop"


def test_an_abstaining_completion_carries_no_citations() -> None:
    completion = make_completion("INSUFFICIENT_CONTEXT")
    result = build_answer(
        "who won the 1998 cup?", "custom-800-80", 5, RETRIEVED, completion, 0.1, "test-embed"
    )

    assert result.abstained is True
    assert result.citations == []
    assert result.invalid_references == []
    assert result.answer == "INSUFFICIENT_CONTEXT"


def test_ungrounded_references_are_flagged_on_the_answer() -> None:
    completion = make_completion("A claim from nowhere [9].")
    result = build_answer("q", "custom-800-80", 2, RETRIEVED, completion, 0.1, "test-embed")

    assert result.abstained is False
    assert result.citations == []
    assert result.invalid_references == [9]


def test_a_missing_reported_model_falls_back_to_empty() -> None:
    completion = make_completion("answer [1]", model=None)
    result = build_answer("q", "spec", 1, RETRIEVED, completion, 0.0, "embed")

    assert result.model == ""

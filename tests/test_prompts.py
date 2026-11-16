"""Prompt construction and abstention detection."""

from __future__ import annotations

from conftest import make_chunk
from ragbench.generation import ABSTENTION_MARKER, build_answer_prompt, detect_abstention


def test_prompt_numbers_passages_and_includes_the_question() -> None:
    retrieved = [
        make_chunk(1, "report.pdf", 1, "The company targets net zero by 2040."),
        make_chunk(2, "report.pdf", 2, "Water consumption fell by 18 percent."),
    ]
    messages = build_answer_prompt("What is the net zero year?", retrieved)

    assert [message["role"] for message in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "What is the net zero year?" in user
    assert "[1] report.pdf (page 1)" in user
    assert "[2] report.pdf (page 2)" in user
    assert "The company targets net zero by 2040." in user
    assert "Water consumption fell by 18 percent." in user
    assert user.index("[1]") < user.index("[2]")


def test_system_prompt_states_the_grounding_rules() -> None:
    messages = build_answer_prompt("q", [make_chunk(1, "a.txt", None, "text")])
    system = messages[0]["content"]
    assert "only the numbered passages" in system
    assert "[1]" in system
    assert ABSTENTION_MARKER in system
    assert "Never invent references" in system


def test_passages_without_pages_say_so() -> None:
    messages = build_answer_prompt("q", [make_chunk(1, "notes.txt", None, "plain text")])
    assert "[1] notes.txt (page unavailable)" in messages[1]["content"]


def test_empty_retrieval_still_builds_a_prompt() -> None:
    messages = build_answer_prompt("q", [])
    assert "(no passages were retrieved)" in messages[1]["content"]


def test_abstention_is_detected_from_the_marker_alone() -> None:
    for content in (
        ABSTENTION_MARKER,
        "insufficient_context",
        f"  {ABSTENTION_MARKER.lower()}  ",
        f'"{ABSTENTION_MARKER}"',
        f"{ABSTENTION_MARKER}.",
        f"{ABSTENTION_MARKER}.\n\nThe passages do not mention a target year.",
        f"  \n{ABSTENTION_MARKER}\nThe passages do not mention a target year.",
    ):
        assert detect_abstention(content), content


def test_answers_are_not_mistaken_for_abstention() -> None:
    for content in (
        "The company targets net zero by 2040 [1].",
        f"{ABSTENTION_MARKER} is the marker I would use, but: the target is 2040 [1].",
        f"According to the passages {ABSTENTION_MARKER.lower()} does not appear.",
        "",
        "   ",
    ):
        assert not detect_abstention(content), content

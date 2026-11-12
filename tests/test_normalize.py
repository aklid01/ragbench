"""Text normalisation rules."""

from __future__ import annotations

from ragbench.ingest.normalize import normalize_text


def test_horizontal_whitespace_collapses() -> None:
    assert normalize_text("a    b\t\tc") == "a b c"


def test_line_endings_become_newlines() -> None:
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_unicode_spaces_become_plain_spaces() -> None:
    assert normalize_text("a\u00a0b\u3000c") == "a b c"


def test_control_characters_are_dropped() -> None:
    assert normalize_text("a\x00b\x07c") == "abc"


def test_soft_hyphen_is_removed() -> None:
    assert normalize_text("sustain\u00adable") == "sustainable"


def test_blank_runs_collapse_to_one_blank_line() -> None:
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_word_broken_across_lines_is_rejoined() -> None:
    assert normalize_text("sus-\ntainable approach") == "sustainable approach"


def test_hyphenated_compound_is_left_alone() -> None:
    assert normalize_text("Anti-\nInflation measures") == "Anti-\nInflation measures"


def test_normalisation_is_idempotent() -> None:
    text = "Sustai-\nnable   goals\u00a0and\u00adtargets\n\n\n\nmore text"
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_empty_input_stays_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""

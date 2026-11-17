"""Citation extraction and validation against retrieved passages."""

from __future__ import annotations

from conftest import make_chunk
from ragbench.generation import build_citations, extract_references

RETRIEVED = [
    make_chunk(1, "report.pdf", 1, "The company targets net zero emissions by 2040."),
    make_chunk(2, "report.pdf", 2, "Water consumption fell by 18 percent in 2021."),
    make_chunk(3, "memo.txt", None, "The audit found two control weaknesses."),
]


def test_references_are_extracted_in_order_with_duplicates() -> None:
    assert extract_references("a [1] then [2][3] then [1] again [2, 3] end") == [
        1,
        2,
        3,
        1,
        2,
        3,
    ]


def test_bracketed_non_numbers_are_ignored() -> None:
    assert extract_references("see [appendix a] and [0x1f] but [2]") == [2]


def test_citations_map_to_the_retrieved_passages() -> None:
    answer = "The target is 2040 [1], and water use fell 18 percent [2]."
    citations, invalid = build_citations(answer, RETRIEVED)

    assert invalid == []
    assert [citation.reference for citation in citations] == [1, 2]
    first = citations[0]
    assert first.chunk_id == "report.pdf-chunk-1"
    assert first.doc_id == "report.pdf-doc"
    assert first.file_name == "report.pdf"
    assert first.page_start == 1
    assert first.page_end == 1
    assert first.citation_label == "report.pdf (page 1)"
    assert "net zero" in first.excerpt
    assert first.claims == ["The target is 2040 [1], and water use fell 18 percent [2]."]


def test_a_claim_keeps_only_its_own_sentence() -> None:
    answer = "The target is 2040 [1]. Water use fell 18 percent [2]."
    citations, _ = build_citations(answer, RETRIEVED)

    assert citations[0].claims == ["The target is 2040 [1]."]
    assert citations[1].claims == ["Water use fell 18 percent [2]."]


def test_invalid_references_are_reported_not_silently_kept() -> None:
    answer = "Invented [7] and also [0] and [99, 2]."
    citations, invalid = build_citations(answer, RETRIEVED)

    assert invalid == [7, 0, 99]
    assert [citation.reference for citation in citations] == [2]
    assert citations[0].claims == ["Invented [7] and also [0] and [99, 2]."]


def test_repeated_references_are_merged_into_one_citation() -> None:
    answer = "First fact [3]. Second fact [3] again."
    citations, invalid = build_citations(answer, RETRIEVED)

    assert invalid == []
    assert len(citations) == 1
    assert citations[0].reference == 3
    assert len(citations[0].claims) == 2


def test_citations_without_pages_mark_them_unavailable() -> None:
    answer = "The audit found weaknesses [3]."
    citations, invalid = build_citations(answer, RETRIEVED)

    assert invalid == []
    assert citations[0].page_start is None
    assert citations[0].citation_label == "memo.txt (page unavailable)"


def test_long_passages_are_truncated_in_the_excerpt() -> None:
    chunk = make_chunk(1, "long.txt", 1, "x" * 400)
    citations, invalid = build_citations("claim [1]", [chunk])

    assert invalid == []
    assert citations[0].excerpt.endswith("...")
    assert len(citations[0].excerpt) == 283


def test_answer_without_references_yields_no_citations() -> None:
    citations, invalid = build_citations("A bare answer with no references.", RETRIEVED)
    assert citations == []
    assert invalid == []

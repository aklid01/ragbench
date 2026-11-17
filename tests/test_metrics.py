"""Metric calculations against hand computed values.

The numbers here are worked out by hand in the comments, so a regression
means the metric definition drifted, not just the code.
"""

from __future__ import annotations

from conftest import make_case_record
from ragbench.evaluation.metrics import (
    GenerationMetrics,
    RetrievalMetrics,
    compute_faithfulness,
    compute_token_f1,
    contains_expected_text,
    mrr_at_k,
    recall_at_k,
)
from ragbench.evaluation.records import CaseRecord


def _record(case_id: str, rank: int | None, **overrides) -> CaseRecord:
    return make_case_record(case_id, rank_of_expected=rank, **overrides)


def test_recall_counts_only_cases_with_a_known_passage() -> None:
    # Three scored cases: ranks 1, 4 and None (never retrieved).
    records = [_record("a", 1), _record("b", 4), _record("c", None)]
    assert recall_at_k(records, 1) == 1 / 3
    assert recall_at_k(records, 3) == 1 / 3
    assert recall_at_k(records, 4) == 2 / 3


def test_mrr_uses_reciprocal_rank_over_every_scored_case() -> None:
    # Ranks 1 and 3, one never retrieved: (1/1 + 1/3) / 3.
    records = [_record("a", 1), _record("b", 3), _record("c", None)]
    assert mrr_at_k(records, 3) == (1.0 + 1 / 3) / 3


def test_mrr_ignores_hits_beyond_k() -> None:
    records = [_record("a", 5), _record("b", 2)]
    assert mrr_at_k(records, 3) == (1 / 2) / 2


def test_errors_and_unanswerable_cases_are_excluded() -> None:
    records = [
        _record("a", 1),
        _record("b", None, error="generation failed"),
        _record("c", None, answerable=False),
    ]
    assert recall_at_k(records, 1) == 1.0
    assert mrr_at_k(records, 3) == 1.0


def test_retrieval_metrics_class_reports_every_k() -> None:
    records = [_record("a", 2), _record("b", 3)]
    metrics = RetrievalMetrics(records, ks=(1, 2, 3))
    assert metrics.recall == {1: 0.0, 2: 0.5, 3: 1.0}
    assert metrics.mrr[1] == 0.0
    # At k=2 only rank 2 counts: (1/2) / 2 scored cases.
    assert metrics.mrr[2] == 0.25
    assert abs(metrics.mrr[3] - (0.5 + 1 / 3) / 2) < 1e-9


def test_contains_expected_text_allows_boundary_loss() -> None:
    chunk = "The board comprises seven independent members with no financial ties to executives."
    assert contains_expected_text(chunk, "seven independent members with no financial ties")
    assert not contains_expected_text(chunk, "wholly unrelated content about fishing quotas")


def test_contains_expected_text_threshold_rejects_weak_overlap() -> None:
    chunk = "alpha beta gamma delta epsilon zeta eta theta"
    assert not contains_expected_text(chunk, "alpha beta omega sigma kappa lambda mu nu")


def test_token_f1_is_symmetric_word_overlap() -> None:
    # Shared: net, zero. Predicted 4 tokens, gold 4 tokens: P=R=0.5, F1=0.5.
    score = compute_token_f1("net zero by 2040", "the net zero target")
    assert abs(score - 0.5) < 1e-9
    assert compute_token_f1("nothing about that here", "net zero by 2040") == 0.0


def test_faithfulness_counts_sentences_with_valid_references() -> None:
    answer = "The target is 2040 [1]. Water fell 18 percent [2]. Unquoted closing sentence."
    assert compute_faithfulness(answer, {1, 2}) == 2 / 3
    # A sentence citing only invalid references does not count.
    assert compute_faithfulness("Claim one [9]. Claim two [1].", {1}) == 0.5
    # Every sentence cited.
    assert compute_faithfulness("Claim one [1]. Claim two [2][1].", {1, 2}) == 1.0


def test_generation_metrics_abstention_and_citations() -> None:
    good = make_case_record(
        "good",
        rank_of_expected=1,
        answerable=True,
        expected_text="target 2040 net zero emissions",
        cited_expected=True,
        cited_expected_references=[1],
        abstained=False,
        citations=[
            {
                "reference": 1,
                "chunk_id": "c1",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 1,
                "page_end": 1,
                "excerpt": "target 2040 net zero emissions",
                "claims": ["s [1]."],
            }
        ],
        retrieved=[
            {
                "rank": 1,
                "chunk_id": "c1",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "target 2040 net zero emissions",
            }
        ],
    )
    abstained_unanswerable = make_case_record("unans", answerable=False, abstained=True)
    answered_unanswerable = make_case_record("unans2", answerable=False, abstained=False)
    metrics = GenerationMetrics([good, abstained_unanswerable, answered_unanswerable])
    assert metrics.abstention_rate_unanswerable == 0.5
    assert metrics.answered_answerable_rate == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0


def test_citation_precision_counts_citations_not_cases() -> None:
    # One scored case, 2 citations, both pointing at the expected passage:
    # precision must be 2/2 = 1.0. A record-level numerator over a
    # citation-level denominator would report 1/2 = 0.5 here.
    record = make_case_record(
        "a",
        rank_of_expected=1,
        answerable=True,
        expected_text="target 2040 net zero emissions",
        cited_expected=True,
        cited_expected_references=[1, 2],
        abstained=False,
        citations=[
            {
                "reference": 1,
                "chunk_id": "c1",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 1,
                "page_end": 1,
                "excerpt": "target 2040 net zero emissions",
                "claims": ["s1 [1]."],
            },
            {
                "reference": 2,
                "chunk_id": "c2",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 1,
                "page_end": 1,
                "excerpt": "target 2040 net zero emissions",
                "claims": ["s2 [2]."],
            },
        ],
    )
    metrics = GenerationMetrics([record])
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0


def test_citation_precision_penalises_wrong_citations_per_citation() -> None:
    # Two cases, 3 citations total, 2 point at expected passages: 2/3.
    correct = make_case_record(
        "a",
        answerable=True,
        expected_text="target 2040 net zero emissions",
        cited_expected=True,
        cited_expected_references=[1, 2],
        abstained=False,
        citations=[
            {
                "reference": 1,
                "chunk_id": "c1",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 1,
                "page_end": 1,
                "excerpt": "target 2040 net zero emissions",
                "claims": ["s1 [1]."],
            },
            {
                "reference": 2,
                "chunk_id": "c2",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 1,
                "page_end": 1,
                "excerpt": "target 2040 net zero emissions",
                "claims": ["s2 [2]."],
            },
        ],
    )
    wrong = make_case_record(
        "b",
        answerable=True,
        expected_text="water use fell 18 percent",
        cited_expected=False,
        cited_expected_references=[],
        abstained=False,
        citations=[
            {
                "reference": 1,
                "chunk_id": "c3",
                "doc_id": "d",
                "file_name": "f.pdf",
                "page_start": 2,
                "page_end": 2,
                "excerpt": "unrelated passage",
                "claims": ["s3 [1]."],
            },
        ],
    )
    metrics = GenerationMetrics([correct, wrong])
    assert metrics.citation_precision == 2 / 3
    # Recall is still per case: one of two cases cited the expected passage.
    assert metrics.citation_recall == 0.5

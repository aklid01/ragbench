"""Metric calculations over case records.

Every metric is a pure function over :class:`CaseRecord` sequences, so a
number in a run file can always be recomputed from the records next to it.

Design decisions the numbers depend on:

- Retrieval credit uses the case's ``expected_text``: a chunk counts as the
  expected passage when roughly 80 percent of the expected tokens appear in
  the chunk in order (chunk boundaries cut passages, so exact containment is
  too strict).
- ``token_f1`` is the token level F1 between the answer and the reference
  answer, a standard transparent lexical comparison, not a judgment of
  truth. The reference answers in the dataset remain the source of truth.
- ``faithfulness`` is the share of answer sentences that carry a validated
  citation, a structural check by the citation layer, not a claim that the
  cited passage semantically supports the sentence.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections.abc import Sequence

from ..generation import extract_references
from ..text import split_sentences
from .records import CaseRecord

__all__ = [
    "MATCH_THRESHOLD",
    "token_set",
    "contains_expected_text",
    "compute_token_f1",
    "compute_exact_match",
    "compute_faithfulness",
    "recall_at_k",
    "mrr_at_k",
    "RetrievalMetrics",
    "GenerationMetrics",
    "OperationalMetrics",
]

MATCH_THRESHOLD = 0.8

_WORD = re.compile(r"[a-z0-9]+")


def token_set(text: str) -> list[str]:
    """Lower cased, accent stripped word tokens of ``text``."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _WORD.findall(stripped)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token sequences."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[index - 1], previous[index]))
        previous = current
    return previous[-1]


def contains_expected_text(chunk_text: str, expected_text: str) -> bool:
    """True when the chunk holds enough of the expected passage.

    The share of expected tokens found in order inside the chunk must reach
    :data:`MATCH_THRESHOLD`. Subsequence, not contiguous, because page
    headers and footers interleave with report prose.
    """
    expected = token_set(expected_text)
    if not expected:
        return False
    chunk = token_set(chunk_text)
    matched = _lcs_length(chunk, expected)
    return matched / len(expected) >= MATCH_THRESHOLD


def compute_token_f1(answer: str, reference: str) -> float:
    """Token level F1 between an answer and its reference answer."""
    predicted = token_set(answer)
    gold = token_set(reference)
    if not predicted or not gold:
        return 0.0
    common = {}
    for token in predicted:
        common[token] = min(predicted.count(token), gold.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(answer: str, reference: str) -> bool:
    """True when the answer equals the reference after aggressive normalisation."""
    return token_set(answer) == token_set(reference)


def compute_faithfulness(answer: str, valid_references: set[int]) -> float:
    """Share of answer sentences that contain at least one valid reference.

    An answer with no sentences scores 0.0. A sentence citing only invalid
    references counts as unfaithful.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        references = extract_references(sentence)
        if references and any(reference in valid_references for reference in references):
            supported += 1
    return supported / len(sentences)


def _scored(outcomes: Sequence[CaseRecord]) -> list[CaseRecord]:
    """Answerable, error free records with a known expected passage."""
    return [
        record
        for record in outcomes
        if record.answerable and record.error is None and record.expected_text
    ]


def recall_at_k(outcomes: Sequence[CaseRecord], k: int) -> float:
    """Share of scored answerable cases whose expected passage appeared in top-K."""
    scored = _scored(outcomes)
    if not scored:
        return 0.0
    hits = sum(
        1
        for record in scored
        if record.rank_of_expected is not None and record.rank_of_expected <= k
    )
    return hits / len(scored)


def mrr_at_k(outcomes: Sequence[CaseRecord], k: int) -> float:
    """Mean reciprocal rank of the expected passage, capped at rank K.

    Cases whose passage was never retrieved contribute 0 to the mean, which
    is why the denominator is every scored case, not only the hits.
    """
    scored = _scored(outcomes)
    if not scored:
        return 0.0
    total = sum(
        1.0 / record.rank_of_expected
        for record in scored
        if record.rank_of_expected is not None and record.rank_of_expected <= k
    )
    return total / len(scored)


class RetrievalMetrics:
    """Recall@K and MRR@K at the requested depths."""

    def __init__(self, outcomes: Sequence[CaseRecord], ks: Sequence[int] = (3, 5, 8)) -> None:
        self.recall = {k: recall_at_k(outcomes, k) for k in ks}
        self.mrr = {k: mrr_at_k(outcomes, k) for k in ks}

    def as_dict(self) -> dict[str, dict[str, float]]:
        recall = {str(k): v for k, v in self.recall.items()}
        mrr = {str(k): v for k, v in self.mrr.items()}
        return {"recall": recall, "mrr": mrr}


class GenerationMetrics:
    """Generation quality from the records, abstention scored separately.

    ``citation_precision`` is the share of validated citations that point at
    the expected passage, over scored answerable cases. ``citation_recall``
    is the share of scored cases with at least one citation to the expected
    passage. Both are ``None`` when no scored case produced any citation.
    """

    def __init__(self, outcomes: Sequence[CaseRecord]) -> None:
        scored = _scored(outcomes)
        unanswerable = [
            record for record in outcomes if not record.answerable and record.error is None
        ]

        self.abstention_rate_unanswerable = (
            sum(1 for record in unanswerable if record.abstained) / len(unanswerable)
            if unanswerable
            else None
        )
        self.answered_answerable_rate = (
            sum(1 for record in scored if not record.abstained) / len(scored) if scored else None
        )

        citation_count = sum(len(record.citations) for record in scored)
        cited_expected = sum(1 for record in scored if record.cited_expected)
        self.citation_precision = cited_expected / citation_count if citation_count else None
        self.citation_recall = cited_expected / len(scored) if scored else None

        invalid_answerable = [record for record in scored if not record.abstained]
        self.invalid_reference_rate = (
            sum(1 for record in invalid_answerable if record.invalid_references)
            / len(invalid_answerable)
            if invalid_answerable
            else None
        )

        faithful = [
            record.faithfulness for record in invalid_answerable if record.faithfulness is not None
        ]
        self.faithfulness_mean = statistics.mean(faithful) if faithful else None

        f1_scores = [record.token_f1 for record in scored if record.token_f1 is not None]
        self.token_f1_mean = statistics.mean(f1_scores) if f1_scores else None
        self.exact_match_rate = (
            sum(1 for record in scored if record.exact_match) / len(scored) if scored else None
        )

    def as_dict(self) -> dict[str, float | None]:
        return {
            "abstention_rate_unanswerable": self.abstention_rate_unanswerable,
            "answered_answerable_rate": self.answered_answerable_rate,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "invalid_reference_rate": self.invalid_reference_rate,
            "faithfulness_mean": self.faithfulness_mean,
            "token_f1_mean": self.token_f1_mean,
            "exact_match_rate": self.exact_match_rate,
        }


class OperationalMetrics:
    """Run level operational accounting."""

    def __init__(self, outcomes: Sequence[CaseRecord]) -> None:
        scored = [record for record in outcomes if record.error is None]
        tokens = [record.total_tokens for record in scored if record.total_tokens is not None]
        retrieval = [record.retrieval_seconds for record in scored]
        generation = [record.generation_seconds for record in scored]
        self.total_cases = len(outcomes)
        self.errors = len(outcomes) - len(scored)
        self.mean_retrieval_seconds = statistics.mean(retrieval) if retrieval else 0.0
        self.mean_generation_seconds = statistics.mean(generation) if generation else 0.0
        self.total_tokens = sum(tokens) if tokens else None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "total_cases": self.total_cases,
            "errors": self.errors,
            "mean_retrieval_seconds": round(self.mean_retrieval_seconds, 3),
            "mean_generation_seconds": round(self.mean_generation_seconds, 3),
            "total_tokens": self.total_tokens,
        }

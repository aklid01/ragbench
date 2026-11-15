"""Citation extraction and validation.

A reference is only a citation when its number maps back to a passage that
was actually retrieved; anything else (an invented reference, a gap in the
numbering) is reported as invalid instead of being silently accepted. Each
citation carries the sentence(s) it supports and an excerpt of the cited
passage, so a human reviewer can check support without opening the corpus.
Structural validity is all this layer claims: whether the passage truly
backs the claim is a judgment for the evaluation phase.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field, computed_field

from ..models import citation_label_for
from ..retrieval.search import RetrievedChunk

__all__ = ["EXCERPT_CHAR_LIMIT", "Citation", "build_citations", "extract_references"]

EXCERPT_CHAR_LIMIT = 280

_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class Citation(BaseModel):
    """One validated reference to a retrieved passage."""

    reference: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    excerpt: str = Field(description="the cited passage, truncated to EXCERPT_CHAR_LIMIT")
    claims: list[str] = Field(description="the answer sentences this reference supports")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation_label(self) -> str:
        return citation_label_for(self.file_name, self.page_start, self.page_end)


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHAR_LIMIT:
        return text
    return text[:EXCERPT_CHAR_LIMIT] + "..."


def extract_references(text: str) -> list[int]:
    """Every bracketed reference in order of appearance, duplicates kept."""
    numbers: list[int] = []
    for match in _CITATION_PATTERN.finditer(text):
        for raw in match.group(1).split(","):
            numbers.append(int(raw))
    return numbers


def build_citations(
    answer: str, retrieved: Sequence[RetrievedChunk]
) -> tuple[list[Citation], list[int]]:
    """Validate the answer's references against the retrieved passages.

    Returns the valid citations in order of first appearance and the invalid
    reference numbers that point at nothing that was retrieved.
    """
    claims_by_reference: dict[int, list[str]] = {}
    for sentence in _sentences(answer):
        for reference in extract_references(sentence):
            claims_by_reference.setdefault(reference, []).append(sentence)

    citations: list[Citation] = []
    invalid: list[int] = []
    for reference, claims in claims_by_reference.items():
        if not 1 <= reference <= len(retrieved):
            invalid.append(reference)
            continue
        chunk = retrieved[reference - 1]
        citations.append(
            Citation(
                reference=reference,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                file_name=chunk.file_name,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                excerpt=_excerpt(chunk.text),
                claims=claims,
            )
        )
    return citations, invalid


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]

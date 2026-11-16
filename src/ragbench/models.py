"""Core data models.

Every model carries the provenance needed to answer "where did this come
from". ``page_number`` is ``None`` for formats without pages (text, markdown)
rather than faked with a page 1, and the citation layer renders it as
unavailable.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "MediaType",
    "SourceRef",
    "Page",
    "ExtractedDocument",
    "Chunk",
    "doc_id_for",
    "content_hash",
    "citation_label_for",
]

MediaType = Literal["pdf", "text", "markdown"]

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def content_hash(text: str) -> str:
    """Stable short hash of a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def doc_id_for(file_name: str, sha256: str) -> str:
    """Build a readable document id: name slug plus digest suffix."""
    stem = Path(file_name).stem.lower()
    slug = _SLUG_PATTERN.sub("-", stem).strip("-") or "document"
    return f"{slug}-{sha256[:8]}"


class SourceRef(BaseModel):
    """Identity of the file a piece of text came from."""

    path: str
    file_name: str
    sha256: str
    media_type: MediaType
    byte_size: int
    ingested_at: datetime

    def short(self) -> str:
        return f"{self.file_name}"


class Page(BaseModel):
    """One page of extracted text. Empty text means the page had nothing extractable."""

    page_number: int | None
    text: str
    char_count: int = Field(
        default=0,
        description="character count of the page text, computed from text when omitted or 0",
    )

    def model_post_init(self, _context: object) -> None:
        if not self.char_count:
            object.__setattr__(self, "char_count", len(self.text))


class ExtractedDocument(BaseModel):
    """A document after extraction, before chunking."""

    doc_id: str
    title: str
    source: SourceRef
    pages: list[Page] = Field(default_factory=list)
    extractor: str
    warnings: list[str] = Field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def has_pages(self) -> bool:
        return any(page.page_number is not None for page in self.pages)


def citation_label_for(file_name: str, page_start: int | None, page_end: int | None) -> str:
    """Human readable provenance shared by chunks, retrieved hits and citations."""
    if page_start is None:
        return f"{file_name} (page unavailable)"
    if page_start == page_end:
        return f"{file_name} (page {page_start})"
    return f"{file_name} (pages {page_start}-{page_end})"


class Chunk(BaseModel):
    """A retrievable slice of a document."""

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    content_hash: str
    source: SourceRef

    @property
    def citation_label(self) -> str:
        """Human readable provenance, used by the citation layer."""
        return citation_label_for(self.source.file_name, self.page_start, self.page_end)

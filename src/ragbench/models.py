"""Core data models.

Three things flow through the pipeline: a source file, the pages extracted
from it, and the chunks cut out of those pages. Every one of them carries the
provenance needed to answer "where did this come from", because a RAG system
that cannot name its source cannot be checked.

Page numbers are optional on purpose. A PDF has real pages; a plain text file
has one continuous stream that was never paginated. Claiming ``page 1`` for a
text file would be a small lie that later has to be explained in every
citation, so ``None`` means "this format has no pages" and the citation layer
renders it as unavailable.
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
]

MediaType = Literal["pdf", "text", "markdown"]

_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def content_hash(text: str) -> str:
    """Stable short hash of a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def doc_id_for(file_name: str, sha256: str) -> str:
    """Build a readable, collision resistant document id.

    The slug keeps the identifier legible in logs and citations, the digest
    suffix keeps two files with the same name apart.
    """
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
    """One page of extracted text.

    ``page_number`` is 1-based for paginated formats and ``None`` otherwise.
    ``text`` is stored after normalisation, so an empty page means the page
    really had no extractable text (a cover image, a divider) rather than that
    the extractor failed.
    """

    page_number: int | None
    text: str
    char_count: int = 0

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
        if self.page_start is None:
            return f"{self.source.file_name} (page unavailable)"
        if self.page_start == self.page_end:
            return f"{self.source.file_name} (page {self.page_start})"
        return f"{self.source.file_name} (pages {self.page_start}-{self.page_end})"

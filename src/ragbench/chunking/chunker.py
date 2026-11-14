"""Token window chunking.

Pages are joined with a page offset map, the joined text is tokenised once,
and a window of ``chunk_size_tokens`` slides over it with a step of
``size - overlap``, so consecutive chunks share exactly the overlap. Windows
are decoded back to text, located by offset and trimmed, giving exact
character and page provenance. Whitespace-only windows produce no chunk and
identical input always produces identical chunk ids and offsets.
"""

from __future__ import annotations

from bisect import bisect_right

from ..errors import ChunkingError
from ..models import Chunk, ExtractedDocument, Page, content_hash
from .spec import ChunkingSpec
from .tokenizer import TokenCounter

__all__ = ["PAGE_SEPARATOR", "join_pages", "chunk_document", "chunk_id_for"]

PAGE_SEPARATOR = "\n\n"


def chunk_id_for(doc_id: str, ordinal: int) -> str:
    """Deterministic id of the ``ordinal``-th chunk of a document."""
    return f"{doc_id}#c{ordinal:04d}"


def join_pages(pages: list[Page]) -> tuple[str, list[tuple[int, int, int | None]]]:
    """Join page texts and return the text with its page spans.

    Each span is ``(start, end, page_number)`` with an exclusive end. The
    separator between two pages belongs to neither of them.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int | None]] = []
    offset = 0
    for index, page in enumerate(pages):
        parts.append(page.text)
        spans.append((offset, offset + len(page.text), page.page_number))
        offset += len(page.text)
        if index < len(pages) - 1:
            parts.append(PAGE_SEPARATOR)
            offset += len(PAGE_SEPARATOR)
    return "".join(parts), spans


def _page_at(
    spans: list[tuple[int, int, int | None]], starts: list[int], offset: int
) -> int | None:
    index = bisect_right(starts, offset) - 1
    if index < 0:
        return None
    return spans[index][2]


def chunk_document(
    document: ExtractedDocument,
    spec: ChunkingSpec,
    counter: TokenCounter,
) -> list[Chunk]:
    """Cut ``document`` into chunks according to ``spec``."""
    spec.validate()
    pages = [page for page in document.pages if page.text.strip()]
    if not pages:
        return []

    text, spans = join_pages(pages)
    starts = [span[0] for span in spans]
    tokens = counter.encode(text)
    if not tokens:
        return []

    size = spec.chunk_size_tokens
    step = size - spec.chunk_overlap_tokens
    chunks: list[Chunk] = []
    cursor = 0
    start_token = 0
    ordinal = 0

    while start_token < len(tokens):
        end_token = min(start_token + size, len(tokens))
        window_text = counter.decode(tokens[start_token:end_token])

        if window_text.strip():
            char_start = text.find(window_text, cursor)
            if char_start < 0:
                char_start = text.find(window_text)
            if char_start < 0:
                raise ChunkingError(
                    f"decoded window for document {document.doc_id} could not be located "
                    "in the document text; the tokenizer round trip is not exact"
                )
            char_end = char_start + len(window_text)
            stripped = window_text.strip()
            leading = len(window_text) - len(window_text.lstrip())
            trailing = len(window_text) - len(window_text.rstrip())
            chunk_start = char_start + leading
            chunk_end = char_end - trailing

            chunks.append(
                Chunk(
                    chunk_id=chunk_id_for(document.doc_id, ordinal),
                    doc_id=document.doc_id,
                    ordinal=ordinal,
                    text=stripped,
                    token_count=counter.count(stripped),
                    char_start=chunk_start,
                    char_end=chunk_end,
                    page_start=_page_at(spans, starts, chunk_start),
                    page_end=_page_at(spans, starts, chunk_end - 1),
                    content_hash=content_hash(stripped),
                    source=document.source,
                )
            )
            ordinal += 1
            cursor = char_end

        start_token += step

    return chunks

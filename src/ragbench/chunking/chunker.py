"""Token window chunking.

Pages are joined with a page offset map, the joined text is tokenised once,
and a window of ``chunk_size_tokens`` slides over it with a step of
``size - overlap``, so consecutive chunks share exactly the overlap. Token
boundaries are mapped back to character offsets through the tokens' UTF-8
byte lengths and the text is sliced directly, which stays exact even when a
window boundary falls inside a multi byte character (decoding such a window
would emit a replacement character that no longer matches the source). A
boundary inside a character snaps outward: window starts move to the next
character edge, window ends to the previous one, so every chunk is an exact
substring of the document text. Whitespace-only windows produce no chunk and
identical input always produces identical chunk ids and offsets.
"""

from __future__ import annotations

from bisect import bisect_right

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


def _char_byte_map(text: str) -> list[int]:
    """Cumulative UTF-8 byte offset of every character boundary."""
    offsets = [0] * (len(text) + 1)
    total = 0
    for index, char in enumerate(text):
        total += len(char.encode("utf-8"))
        offsets[index + 1] = total
    return offsets


def _byte_to_char(map: list[int], offset: int, *, round_forward: bool) -> int:
    """Character index for a byte offset, snapping outward at character edges.

    A byte offset that lands inside a multi byte character resolves to the
    character after it when a window starts there and the character before
    it when a window ends there, so slices never carry a partial character.
    """
    index = bisect_right(map, offset) - 1
    if map[index] == offset:
        return index
    return index + 1 if round_forward else index


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

    byte_at = [0]
    for length in counter.byte_lengths(tokens):
        byte_at.append(byte_at[-1] + length)
    char_byte = _char_byte_map(text)

    size = spec.chunk_size_tokens
    step = size - spec.chunk_overlap_tokens
    chunks: list[Chunk] = []
    ordinal = 0
    start_token = 0

    while start_token < len(tokens):
        end_token = min(start_token + size, len(tokens))
        char_start = _byte_to_char(char_byte, byte_at[start_token], round_forward=True)
        char_end = _byte_to_char(char_byte, byte_at[end_token], round_forward=False)
        if char_end <= char_start:
            start_token += step
            continue

        window = text[char_start:char_end]
        stripped = window.strip()
        if stripped:
            leading = len(window) - len(window.lstrip())
            trailing = len(window) - len(window.rstrip())
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

        start_token += step

    return chunks

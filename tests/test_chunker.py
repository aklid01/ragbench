"""Chunking: boundaries, overlap, provenance and coverage."""

from __future__ import annotations

import pytest

from conftest import make_document
from ragbench.chunking.chunker import chunk_document, join_pages
from ragbench.chunking.spec import ChunkingSpec
from ragbench.errors import ChunkingError
from ragbench.models import Page

SPEC = ChunkingSpec(name="test", chunk_size_tokens=40, chunk_overlap_tokens=8)


def long_document(pages: int = 3, words_per_page: int = 60):
    """A document whose pages carry enough words to need several chunks."""
    document = make_document()
    document.pages = [
        Page(
            page_number=index + 1,
            text=" ".join(f"w{index}-{position}" for position in range(words_per_page)),
        )
        for index in range(pages)
    ]
    return document


def _word_ranges(text: str):
    start = None
    for index, character in enumerate(text):
        if not character.isspace():
            if start is None:
                start = index
        elif start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(text)


def test_chunks_respect_the_token_budget(counter) -> None:
    chunks = chunk_document(long_document(), SPEC, counter)
    assert len(chunks) > 1
    assert all(chunk.token_count <= SPEC.chunk_size_tokens for chunk in chunks)


def test_no_empty_chunks_are_produced(counter) -> None:
    chunks = chunk_document(long_document(), SPEC, counter)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_ids_are_unique_and_ordered(counter) -> None:
    chunks = chunk_document(long_document(), SPEC, counter)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert ids[0].endswith("#c0000")


def test_consecutive_chunks_overlap(counter) -> None:
    document = long_document()
    text, _ = join_pages(document.pages)
    chunks = chunk_document(document, SPEC, counter)
    for current, following in zip(chunks, chunks[1:], strict=False):
        shared = text[following.char_start : current.char_end]
        assert counter.count(shared) >= SPEC.chunk_overlap_tokens - 2


def test_chunks_are_exact_substrings_of_the_document(counter) -> None:
    document = long_document()
    text, _ = join_pages(document.pages)
    for chunk in chunk_document(document, SPEC, counter):
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_every_word_survives_the_chunking(counter) -> None:
    document = long_document()
    text, _ = join_pages(document.pages)
    chunks = chunk_document(document, SPEC, counter)
    covered = [False] * len(text)
    for chunk in chunks:
        for position in range(chunk.char_start, chunk.char_end):
            covered[position] = True
    uncovered = {
        text[start:end] for start, end in _word_ranges(text) if not all(covered[start:end])
    }
    assert uncovered == set()


def test_page_numbers_are_preserved(counter) -> None:
    document = long_document(pages=3)
    chunks = chunk_document(document, SPEC, counter)
    assert chunks
    assert all(chunk.page_start is not None for chunk in chunks)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)


def test_documents_without_pages_report_no_page_numbers(counter) -> None:
    document = make_document()
    document.pages = [Page(page_number=None, text=" ".join(f"w{i}" for i in range(120)))]
    chunks = chunk_document(document, SPEC, counter)
    assert chunks
    assert all(chunk.page_start is None for chunk in chunks)
    assert all(chunk.page_end is None for chunk in chunks)


def test_chunking_is_deterministic(counter) -> None:
    document = long_document()
    first = chunk_document(document, SPEC, counter)
    second = chunk_document(document, SPEC, counter)
    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]


def test_empty_document_produces_no_chunks(counter) -> None:
    document = make_document(pages=["", "   "])
    assert chunk_document(document, SPEC, counter) == []


def test_invalid_spec_is_rejected(counter) -> None:
    document = long_document()
    with pytest.raises(ChunkingError):
        chunk_document(document, ChunkingSpec("bad", 10, 10), counter)
    with pytest.raises(ChunkingError):
        chunk_document(document, ChunkingSpec("bad", 0, 0), counter)


def test_window_boundary_inside_a_multibyte_character(counter) -> None:
    """A boundary splitting a multi byte character must not break provenance.

    tiktoken decodes such a window with a replacement character, which no
    longer matches the source; slicing through byte offsets stays exact.
    Regression: the large-1200 preset failed on a report holding private use
    glyphs because a 1200 token window ended inside one.
    """
    glyph = "\uf0fc"  # one private use character, 3 UTF-8 bytes
    filler = "alpha beta gamma delta epsilon "
    text = (filler + glyph) * 120
    document = make_document()
    document.pages = [Page(page_number=1, text=text)]
    spec = ChunkingSpec(name="glyph", chunk_size_tokens=30, chunk_overlap_tokens=3)
    chunks = chunk_document(document, spec, counter)
    assert chunks
    joined, _ = join_pages(document.pages)
    for chunk in chunks:
        assert joined[chunk.char_start : chunk.char_end] == chunk.text
        assert "\ufffd" not in chunk.text
    covered = "".join(chunk.text for chunk in chunks)
    assert glyph in covered

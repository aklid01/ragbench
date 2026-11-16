"""Chunking pipeline: cut extracted documents and write one directory per configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..paths import ProjectPaths
from ..store import list_documents, load_document, write_chunks
from .chunker import chunk_document
from .spec import ChunkingSpec
from .tokenizer import TokenCounter

__all__ = ["ChunkSummary", "chunk_documents"]


@dataclass
class ChunkSummary:
    """What a chunking run produced."""

    spec_name: str
    documents: int = 0
    chunks: int = 0
    token_total: int = 0
    largest_chunk_tokens: int = 0
    failures: list[str] = field(default_factory=list)
    pruned: int = 0

    def describe(self) -> str:
        pruned = f", {self.pruned} pruned" if self.pruned else ""
        return (
            f"{self.chunks} chunks from {self.documents} documents "
            f"({self.token_total} tokens, largest {self.largest_chunk_tokens}{pruned})"
        )


def chunk_documents(
    paths: ProjectPaths,
    spec: ChunkingSpec,
    counter: TokenCounter,
    doc_ids: list[str] | None = None,
) -> ChunkSummary:
    """Chunk the extracted documents and persist the result.

    A full corpus run (``doc_ids=None``) also removes chunk files of
    documents that are no longer extracted, so the configuration on disk
    never holds chunks of retired documents.
    """
    spec.validate()
    summary = ChunkSummary(spec_name=spec.name)
    full_corpus = doc_ids is None
    targets = doc_ids if doc_ids is not None else list_documents(paths)

    for doc_id in targets:
        try:
            document = load_document(paths, doc_id)
        except FileNotFoundError as exc:
            summary.failures.append(str(exc))
            continue
        chunks = chunk_document(document, spec, counter)
        write_chunks(paths, spec.name, doc_id, chunks)
        summary.documents += 1
        summary.chunks += len(chunks)
        for chunk in chunks:
            summary.token_total += chunk.token_count
            summary.largest_chunk_tokens = max(summary.largest_chunk_tokens, chunk.token_count)

    if full_corpus:
        summary.pruned = _prune_stale_chunk_files(paths, spec.name, set(targets))
    return summary


def _prune_stale_chunk_files(paths: ProjectPaths, spec_name: str, kept: set[str]) -> int:
    """Remove chunk files of documents that are no longer extracted."""
    pruned = 0
    for path in sorted((paths.chunks_dir / spec_name).glob("*.json")):
        if path.stem not in kept:
            path.unlink()
            pruned += 1
    return pruned

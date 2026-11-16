"""Indexing pipeline: embed chunks once, then keep the index up to date.

Chunks already present in the index, matched by chunk id, are not embedded
again, so re-running over an unchanged corpus costs nothing. A changed corpus
only embeds the chunks it has not seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..config import EmbeddingConfig
from ..embeddings import EmbeddingClient
from ..errors import RetrievalError
from ..models import Chunk
from ..paths import ProjectPaths
from ..store import load_chunks
from .store import VectorIndex

__all__ = ["IndexSummary", "index_chunks", "latest_indexed_spec", "read_index_manifest"]

MANIFEST_NAME = "manifest.json"


@dataclass
class IndexSummary:
    """What an indexing run did."""

    spec_name: str
    embedded: int = 0
    skipped: int = 0
    rejected: int = 0
    vector_count: int = 0
    dimensions: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.vector_count} vectors for configuration {self.spec_name!r} "
            f"({self.embedded} embedded now, {self.skipped} already present, "
            f"{self.rejected} rejected, {self.dimensions} dimensions)"
        )


def index_chunks(
    paths: ProjectPaths,
    embedding: EmbeddingConfig,
    client: EmbeddingClient,
    spec_name: str,
    *,
    rebuild: bool = False,
) -> IndexSummary:
    """Embed the chunks of ``spec_name`` and persist them with their metadata."""
    spec_dir = paths.chunks_dir / spec_name
    chunk_files = sorted(spec_dir.glob("*.json")) if spec_dir.is_dir() else []
    if not chunk_files:
        available = sorted(path.name for path in paths.chunks_dir.glob("*") if path.is_dir())
        hint = f"; available configurations: {', '.join(available)}" if available else ""
        raise RetrievalError(f"no chunks for configuration {spec_name!r}{hint}")

    index = VectorIndex(paths, embedding, spec_name)
    if rebuild:
        index.reset()
    present = index.existing_ids()

    summary = IndexSummary(spec_name=spec_name)
    todo: list[Chunk] = []
    for chunk_file in chunk_files:
        for chunk in load_chunks(paths, spec_name, chunk_file.stem):
            if chunk.chunk_id in present:
                summary.skipped += 1
            else:
                todo.append(chunk)

    batch = client.embed_passages([chunk.text for chunk in todo])
    rejected_indexes = {item.index for item in batch.rejected}
    accepted = [chunk for position, chunk in enumerate(todo) if position not in rejected_indexes]
    index.add(accepted, batch.vectors)
    summary.embedded = len(accepted)
    summary.rejected = len(batch.rejected)
    for item in batch.rejected:
        summary.warnings.append(f"{todo[item.index].chunk_id} was not embedded: {item.message}")

    summary.vector_count = index.count()
    summary.dimensions = embedding.expected_dimensions or (
        len(batch.vectors[0]) if batch.vectors else index.dimensions()
    )
    _write_manifest(paths, summary, embedding.model)
    return summary


def read_index_manifest(paths: ProjectPaths, spec_name: str) -> dict[str, object]:
    """Read the manifest of one indexed configuration."""
    source = paths.index_dir / spec_name / MANIFEST_NAME
    if not source.is_file():
        raise RetrievalError(
            f"no index for configuration {spec_name!r}; "
            f"build it with: ragbench index --spec {spec_name}"
        )
    return json.loads(source.read_text(encoding="utf-8"))


def latest_indexed_spec(paths: ProjectPaths) -> str | None:
    """The configuration indexed most recently, if any."""
    manifests = sorted(
        paths.index_dir.glob(f"*/{MANIFEST_NAME}"), key=lambda path: path.stat().st_mtime
    )
    return manifests[-1].parent.name if manifests else None


def _write_manifest(
    paths: ProjectPaths, summary: IndexSummary, embedding_model: str
) -> None:
    payload = {
        "spec": summary.spec_name,
        "embedding_model": embedding_model,
        "dimensions": summary.dimensions,
        "vector_count": summary.vector_count,
        "embedded_now": summary.embedded,
        "already_present": summary.skipped,
        "rejected": summary.rejected,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    target = paths.index_dir / summary.spec_name / MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

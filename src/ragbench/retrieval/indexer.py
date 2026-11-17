"""Indexing pipeline: embed chunks once, then keep the index up to date.

Chunks already present in the index, matched by chunk id, are not embedded
again, so re-running over an unchanged corpus costs nothing. A changed corpus
only embeds the chunks it has not seen, and vectors whose chunk files are
gone are pruned, so the index never holds chunks the corpus retired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..embeddings import EmbeddingClient
from ..errors import ArtifactError, RetrievalError
from ..models import Chunk
from ..paths import ProjectPaths
from ..store import MANIFEST_NAME, load_chunks, write_json
from .vector_index import VectorIndex

__all__ = ["IndexSummary", "index_chunks", "latest_indexed_spec", "read_index_manifest"]


@dataclass
class IndexSummary:
    """What an indexing run did."""

    spec_name: str
    embedded: int = 0
    skipped: int = 0
    rejected: int = 0
    pruned: int = 0
    vector_count: int = 0
    dimensions: int = 0
    rejected_ids: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        pruned = f", {self.pruned} pruned" if self.pruned else ""
        return (
            f"{self.vector_count} vectors for configuration {self.spec_name!r} "
            f"({self.embedded} embedded now, {self.skipped} already present, "
            f"{self.rejected} rejected{pruned}, {self.dimensions} dimensions)"
        )


def index_chunks(
    paths: ProjectPaths,
    client: EmbeddingClient,
    spec_name: str,
    *,
    rebuild: bool = False,
) -> IndexSummary:
    """Embed the chunks of ``spec_name`` and persist them with their metadata."""
    embedding = client.config
    spec_dir = paths.chunks_dir / spec_name
    if not spec_dir.is_dir():
        available = sorted(path.name for path in paths.chunks_dir.glob("*") if path.is_dir())
        hint = f"; available configurations: {', '.join(available)}" if available else ""
        raise RetrievalError(f"no chunks for configuration {spec_name!r}{hint}")

    index = VectorIndex(paths, embedding, spec_name)
    if rebuild:
        index.reset()
    present = index.existing_ids()

    summary = IndexSummary(spec_name=spec_name)
    todo: list[Chunk] = []
    current_ids: set[str] = set()
    for chunk_file in sorted(spec_dir.glob("*.json")):
        try:
            chunks = load_chunks(paths, spec_name, chunk_file.stem)
        except ArtifactError as exc:
            summary.failures.append(str(exc))
            continue
        for chunk in chunks:
            current_ids.add(chunk.chunk_id)
            if chunk.chunk_id in present:
                summary.skipped += 1
            else:
                todo.append(chunk)

    _reject_oversized_chunks(todo, embedding.max_input_tokens)
    batch = client.embed_passages([chunk.text for chunk in todo])
    rejected_indexes = {item.index for item in batch.rejected}
    accepted = [chunk for position, chunk in enumerate(todo) if position not in rejected_indexes]
    index.add(accepted, batch.vectors)
    summary.embedded = len(accepted)
    summary.rejected = len(batch.rejected)
    summary.rejected_ids = [todo[item.index].chunk_id for item in batch.rejected]
    for item in batch.rejected:
        summary.warnings.append(f"{todo[item.index].chunk_id} was not embedded: {item.message}")

    stale = present - current_ids
    if stale:
        index.delete_ids(sorted(stale))
    summary.pruned = len(stale)

    summary.vector_count = index.count()
    summary.dimensions = embedding.expected_dimensions or (
        len(batch.vectors[0]) if batch.vectors else index.dimensions()
    )
    _write_manifest(paths, summary, embedding.model)
    return summary


def _reject_oversized_chunks(todo: list[Chunk], max_input_tokens: int | None) -> None:
    """Fail before any API call when chunks exceed the embedding input limit.

    Endpoints either truncate overlong inputs silently or refuse them mid-run;
    both poison retrieval, so the configured cap is enforced here first.
    """
    if max_input_tokens is None or not todo:
        return
    oversized = [chunk for chunk in todo if chunk.token_count > max_input_tokens]
    if not oversized:
        return
    largest = max(oversized, key=lambda chunk: chunk.token_count)
    examples = ", ".join(chunk.chunk_id for chunk in oversized[:5])
    raise RetrievalError(
        f"{len(oversized)} chunks exceed the embedding input limit of {max_input_tokens} "
        f"tokens (largest: {largest.chunk_id} at {largest.token_count} tokens; "
        f"examples: {examples}); cut smaller chunks or raise "
        "RAGBENCH_EMBEDDING_MAX_INPUT_TOKENS if the endpoint accepts more"
    )


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
    """The configuration indexed most recently, if any.

    Ordered by the ``updated_at`` stamp inside each manifest, so copying
    artifacts between machines cannot reshuffle the default.
    """
    best: tuple[datetime, float, str] | None = None
    for manifest_path in paths.index_dir.glob(f"*/{MANIFEST_NAME}"):
        key = (
            _manifest_updated_at(manifest_path),
            manifest_path.stat().st_mtime,
            manifest_path.parent.name,
        )
        if best is None or key > best:
            best = key
    return best[2] if best else None


def _manifest_updated_at(manifest_path) -> datetime:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(str(payload["updated_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return datetime.min.replace(tzinfo=UTC)


def _write_manifest(paths: ProjectPaths, summary: IndexSummary, embedding_model: str) -> None:
    payload = {
        "spec": summary.spec_name,
        "embedding_model": embedding_model,
        "dimensions": summary.dimensions,
        "vector_count": summary.vector_count,
        "embedded_now": summary.embedded,
        "already_present": summary.skipped,
        "rejected": summary.rejected,
        "rejected_ids": summary.rejected_ids,
        "pruned": summary.pruned,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    write_json(paths.index_dir / summary.spec_name / MANIFEST_NAME, payload)

"""Similarity retrieval: a query in, the top-K chunks out with their distances."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from ..config import EmbeddingConfig
from ..embeddings import EmbeddingClient
from ..errors import RetrievalError
from ..models import citation_label_for
from ..paths import ProjectPaths
from .indexer import read_index_manifest
from .vector_index import PAGE_UNAVAILABLE, VectorIndex

__all__ = ["RetrievedChunk", "search_chunks"]


class RetrievedChunk(BaseModel):
    """One retrieved chunk with its provenance and similarity information."""

    rank: int
    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    text: str
    distance: float
    similarity: float = Field(description="1 - cosine distance")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation_label(self) -> str:
        return citation_label_for(self.file_name, self.page_start, self.page_end)


def search_chunks(
    paths: ProjectPaths,
    embedding: EmbeddingConfig,
    client: EmbeddingClient,
    spec_name: str,
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Rank the indexed chunks of ``spec_name`` against ``query``."""
    read_index_manifest(paths, spec_name)
    index = VectorIndex(paths, embedding, spec_name)
    if index.count() == 0:
        raise RetrievalError(
            f"the index for {spec_name!r} is empty; "
            f"build it with: ragbench index --spec {spec_name}"
        )
    vector = client.embed_query(query)
    found = index.query(vector, top_k)
    if not found:
        raise RetrievalError(f"the query returned nothing for configuration {spec_name!r}")

    results = []
    for rank, hit in enumerate(found, start=1):
        metadata = hit["metadata"]
        page_start = metadata["page_start"]
        page_end = metadata["page_end"]
        results.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=str(hit["chunk_id"]),
                doc_id=str(metadata["doc_id"]),
                file_name=str(metadata["file_name"]),
                page_start=page_start if page_start != PAGE_UNAVAILABLE else None,
                page_end=page_end if page_end != PAGE_UNAVAILABLE else None,
                text=str(hit["text"]),
                distance=float(hit["distance"]),
                similarity=1.0 - float(hit["distance"]),
            )
        )
    return results

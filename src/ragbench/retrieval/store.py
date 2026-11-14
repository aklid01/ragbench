"""The persistent vector index, one ChromaDB collection per chunking configuration.

The collection records the embedding model and dimensions it was built with;
opening it under a different configuration raises instead of silently mixing
incompatible vectors. Page metadata uses ``-1`` for formats without pages and
is translated back to ``None`` on the way out.
"""

from __future__ import annotations

from ..config import EmbeddingConfig
from ..errors import RetrievalError
from ..models import Chunk
from ..paths import ProjectPaths

__all__ = ["PAGE_UNAVAILABLE", "VectorIndex"]

PAGE_UNAVAILABLE = -1


def _collection_name(spec_name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in spec_name)
    if len(cleaned) < 3:
        cleaned = f"chunks-{cleaned}"
    return f"chunks-{cleaned}"


class VectorIndex:
    """A persistent collection of chunk vectors with a configuration guard."""

    def __init__(self, paths: ProjectPaths, embedding: EmbeddingConfig, spec_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RetrievalError(
                "the 'chromadb' package is required for the vector index "
                "(install with: pip install 'ragbench[index]')"
            ) from exc

        self.spec_name = spec_name
        self._dimensions = embedding.expected_dimensions or 0
        self._client = chromadb.PersistentClient(
            path=str(paths.index_dir / spec_name),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_collection_name(spec_name),
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding.model,
                "dimensions": self._dimensions,
            },
        )
        stored = self._collection.metadata or {}
        stored_model = stored.get("embedding_model")
        stored_dimensions = stored.get("dimensions")
        if stored_model != embedding.model or stored_dimensions != self._dimensions:
            raise RetrievalError(
                f"the index for {spec_name!r} was built with embedding model "
                f"{stored_model!r} at {stored_dimensions} dimensions, but the configuration "
                f"asks for {embedding.model!r} at {self._dimensions}; rebuild it with --rebuild"
            )

    def count(self) -> int:
        return self._collection.count()

    def existing_ids(self) -> set[str]:
        if self._collection.count() == 0:
            return set()
        return set(self._collection.get(include=[])["ids"])

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise RetrievalError(
                f"{len(vectors)} vectors for {len(chunks)} chunks; refusing to index"
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def query(self, vector: list[float], top_k: int) -> list[dict[str, object]]:
        count = self._collection.count()
        if count == 0:
            return []
        response = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        results: list[dict[str, object]] = []
        for chunk_id, document, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
            strict=True,
        ):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return results

    def reset(self) -> None:
        self._client.delete_collection(_collection_name(self.spec_name))
        self._collection = self._client.get_or_create_collection(
            name=_collection_name(self.spec_name),
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self._collection.metadata["embedding_model"],
                "dimensions": self._collection.metadata["dimensions"],
            },
        )

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, object]:
        return {
            "doc_id": chunk.doc_id,
            "file_name": chunk.source.file_name,
            "ordinal": chunk.ordinal,
            "token_count": chunk.token_count,
            "page_start": chunk.page_start if chunk.page_start is not None else PAGE_UNAVAILABLE,
            "page_end": chunk.page_end if chunk.page_end is not None else PAGE_UNAVAILABLE,
            "content_hash": chunk.content_hash,
        }

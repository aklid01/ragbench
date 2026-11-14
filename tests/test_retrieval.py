"""Vector index, indexer and search: the configuration guard, incremental indexing and page provenance."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from conftest import make_document
from fake_embeddings import hash_embedding
from ragbench.chunking.chunker import chunk_document
from ragbench.chunking.spec import ChunkingSpec
from ragbench.chunking.tokenizer import get_counter
from ragbench.config import EmbeddingConfig
from ragbench.embeddings import EmbeddingBatch, RejectedInput
from ragbench.errors import RetrievalError
from ragbench.retrieval import VectorIndex, index_chunks, latest_indexed_spec, read_index_manifest
from ragbench.retrieval.search import search_chunks
from ragbench.store import write_chunks, write_document

DIMENSIONS = 64


def make_embedding_config(model: str = "test-embed", dimensions: int = DIMENSIONS) -> EmbeddingConfig:
    return EmbeddingConfig(
        base_url="http://embedding.test/v1",
        model=model,
        api_key=SecretStr("k"),
        expected_dimensions=dimensions,
    )


class FakeEmbedder:
    """Counts text and query embeddings without touching a network."""

    def __init__(self, reject: set[str] | None = None) -> None:
        self.embedded_texts: list[str] = []
        self.queries: list[str] = []
        self.reject = reject or set()

    def embed_passages(self, texts: list[str]) -> EmbeddingBatch:
        self.embedded_texts.extend(texts)
        batch = EmbeddingBatch()
        for position, text in enumerate(texts):
            if text in self.reject:
                batch.rejected.append(
                    RejectedInput(position, text, "input rejected by content safety screening")
                )
            else:
                batch.vectors.append(hash_embedding(text, DIMENSIONS))
        return batch

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return hash_embedding(text, DIMENSIONS)


def chunked_document(
    paths: Any,
    pages: list[str],
    spec_name: str = "test-30-5",
    doc_id: str = "report-1234abcd",
    file_name: str = "report.pdf",
) -> list:
    spec = ChunkingSpec(
        name=spec_name, chunk_size_tokens=30, chunk_overlap_tokens=5
    )
    counter = get_counter("cl100k_base")
    document = make_document(source_file=file_name, pages=pages)
    document = document.model_copy(update={"doc_id": doc_id})
    write_document(paths, document)
    chunks = chunk_document(document, spec, counter)
    write_chunks(paths, spec_name, doc_id, chunks)
    return chunks


def text_index(
    paths: Any, spec_name: str = "test-30-5", rebuild: bool = False, **config: Any
):
    embedder = FakeEmbedder()
    summary = index_chunks(
        paths, make_embedding_config(**config), embedder, spec_name, rebuild=rebuild
    )
    return summary, embedder


def test_reindexing_an_unchanged_corpus_makes_no_embedding_calls(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight.", "Wind turbines spin in the harbour."])
    summary, embedder = text_index(paths)
    assert summary.embedded > 0

    second, embedder2 = text_index(paths)
    assert second.embedded == 0
    assert second.skipped == summary.embedded
    assert embedder2.embedded_texts == []


def test_indexing_picks_up_only_new_chunks(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight."], doc_id="doc-a")
    text_index(paths)

    chunked_document(
        paths, ["Wind turbines spin in the harbour."], doc_id="doc-b", file_name="wind.pdf"
    )
    second, embedder2 = text_index(paths)
    assert second.embedded > 0
    assert len(embedder2.embedded_texts) == second.embedded
    assert all("Wind" in text for text in embedder2.embedded_texts)


def test_rebuild_embeds_everything_again(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight.", "Wind turbines spin in the harbour."])
    first, _ = text_index(paths)
    second, embedder2 = text_index(paths, rebuild=True)
    assert second.embedded == first.embedded
    assert len(embedder2.embedded_texts) == first.embedded


def test_index_reports_missing_chunks_with_available_configurations(paths) -> None:
    with pytest.raises(RetrievalError) as excinfo:
        text_index(paths)
    assert "no chunks for configuration" in str(excinfo.value)


def test_opening_an_index_with_a_different_model_is_refused(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight."])
    text_index(paths)
    with pytest.raises(RetrievalError) as excinfo:
        VectorIndex(paths, make_embedding_config(model="other-model"), "test-30-5")
    assert "--rebuild" in str(excinfo.value)


def test_opening_an_index_with_different_dimensions_is_refused(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight."])
    text_index(paths)
    with pytest.raises(RetrievalError) as excinfo:
        VectorIndex(paths, make_embedding_config(dimensions=32), "test-30-5")
    assert "32" in str(excinfo.value)


def test_search_ranks_the_matching_chunk_first(paths) -> None:
    chunked_document(
        paths,
        ["Solar panels and inverters convert sunlight into electricity on the factory roof."],
        doc_id="doc-a",
        file_name="solar.pdf",
    )
    chunked_document(
        paths,
        ["Wind turbines spin in the harbour and feed the grid all winter long."],
        doc_id="doc-b",
        file_name="wind.pdf",
    )
    text_index(paths)
    embedder = FakeEmbedder()
    results = search_chunks(
        paths,
        make_embedding_config(),
        embedder,
        "test-30-5",
        "Solar panels and inverters convert sunlight",
        top_k=2,
    )
    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].file_name == "solar.pdf"
    assert "Solar panels" in results[0].text
    assert results[0].page_start == 1
    assert results[0].distance < results[1].distance
    assert "page 1" in results[0].citation_label


def test_search_reports_unavailable_pages_for_unpaginated_chunks(paths) -> None:
    spec = ChunkingSpec(name="md-20-2", chunk_size_tokens=20, chunk_overlap_tokens=2)
    counter = get_counter("cl100k_base")
    document = make_document(source_file="notes.md", pages=["Markdown notes about geothermal heat pumps."])
    object.__setattr__(document.pages[0], "page_number", None)
    write_document(paths, document)
    write_chunks(paths, spec.name, document.doc_id, chunk_document(document, spec, counter))
    index_chunks(paths, make_embedding_config(), FakeEmbedder(), spec.name)

    results = search_chunks(
        paths, make_embedding_config(), FakeEmbedder(), spec.name, "geothermal heat pumps", 1
    )
    assert results[0].page_start is None
    assert "page unavailable" in results[0].citation_label


def test_top_k_larger_than_the_index_returns_everything(paths) -> None:
    chunked_document(paths, ["Solar panels.", "Wind turbines.", "Geothermal pumps.", "Battery storage."])
    text_index(paths)
    results = search_chunks(
        paths, make_embedding_config(), FakeEmbedder(), "test-30-5", "solar panels", top_k=50
    )
    assert len(results) == len({result.chunk_id for result in results})


def test_search_without_an_index_fails_cleanly(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters."])
    with pytest.raises(RetrievalError) as excinfo:
        search_chunks(paths, make_embedding_config(), FakeEmbedder(), "test-30-5", "solar", 5)
    assert "no index for configuration" in str(excinfo.value)


def test_search_on_an_empty_index_fails_cleanly(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters."])
    manifest_dir = paths.index_dir / "test-30-5"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({"spec": "test-30-5"}), encoding="utf-8")
    with pytest.raises(RetrievalError) as excinfo:
        search_chunks(paths, make_embedding_config(), FakeEmbedder(), "test-30-5", "solar", 5)
    assert "empty" in str(excinfo.value)


def test_manifest_records_the_embedding_configuration(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight."])
    text_index(paths)
    manifest = read_index_manifest(paths, "test-30-5")
    assert manifest["embedding_model"] == "test-embed"
    assert manifest["dimensions"] == DIMENSIONS
    assert manifest["vector_count"] > 0


def test_latest_indexed_spec_prefers_the_most_recent_manifest(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight."], spec_name="spec-a")
    text_index(paths, spec_name="spec-a")
    assert latest_indexed_spec(paths) == "spec-a"
    chunked_document(paths, ["Wind turbines spin."], spec_name="spec-b")
    text_index(paths, spec_name="spec-b")
    assert latest_indexed_spec(paths) == "spec-b"


def test_two_configurations_get_separate_indexes(paths) -> None:
    chunked_document(paths, ["Solar panels and inverters convert sunlight."], spec_name="spec-a")
    text_index(paths, spec_name="spec-a")
    chunked_document(paths, ["Wind turbines spin."], spec_name="spec-b")
    text_index(paths, spec_name="spec-b")

    assert read_index_manifest(paths, "spec-a")["vector_count"] > 0
    assert read_index_manifest(paths, "spec-b")["vector_count"] > 0
    results_a = search_chunks(
        paths, make_embedding_config(), FakeEmbedder(), "spec-a", "solar panels", 5
    )
    assert all(result.doc_id.startswith("report-") for result in results_a)


def test_rejected_chunks_are_reported_and_not_indexed(paths) -> None:
    chunked_document(
        paths,
        ["Solar panels and inverters convert sunlight into electricity on the factory roof."],
        doc_id="doc-a",
        file_name="solar.pdf",
    )
    chunked_document(
        paths,
        ["Wind turbines spin in the harbour and feed the grid all winter long."],
        doc_id="doc-b",
        file_name="wind.pdf",
    )
    embedder = FakeEmbedder(reject={"Wind turbines spin in the harbour and feed the grid all winter long."})
    summary = index_chunks(paths, make_embedding_config(), embedder, "test-30-5")

    assert summary.embedded == 1
    assert summary.rejected == 1
    assert summary.warnings and "content safety" in summary.warnings[0]
    assert summary.vector_count == 1

    manifest = read_index_manifest(paths, "test-30-5")
    assert manifest["rejected"] == 1

    results = search_chunks(
        paths, make_embedding_config(), FakeEmbedder(), "test-30-5", "wind turbines", 5
    )
    assert all("wind" not in result.text.lower() for result in results)

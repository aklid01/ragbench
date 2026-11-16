"""End to end: a corpus goes in, an index is built, a query ranks the right chunk first.

The test drives the real command line entry point against a fake
OpenAI-compatible embeddings server on localhost, so configuration loading,
HTTP requests, vector persistence and retrieval are all exercised together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_pdf, write_text
from fake_embeddings import FakeEmbeddingServer
from ragbench.cli import main
from ragbench.paths import ProjectPaths

DIMENSIONS = 64


@pytest.fixture
def embeddings(monkeypatch: pytest.MonkeyPatch):
    """A fake embeddings server wired into the process environment."""
    server = FakeEmbeddingServer()
    monkeypatch.setenv("RAGBENCH_EMBEDDING_BASE_URL", server.base_url)
    monkeypatch.setenv("RAGBENCH_EMBEDDING_API_KEY", "test-key")
    monkeypatch.setenv("RAGBENCH_EMBEDDING_DIMENSIONS", str(DIMENSIONS))
    monkeypatch.setenv("RAGBENCH_EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("RAGBENCH_EMBEDDING_MAX_RETRIES", "1")
    monkeypatch.setenv("RAGBENCH_REQUESTS_PER_MINUTE", "100000")
    yield server
    server.stop()


def _build_corpus(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_pdf(
        directory / "report.pdf",
        [
            "The company targets net zero emissions by 2040.",
            "Water consumption fell by 18 percent in 2021.",
            "The board comprises seven independent directors.",
        ],
    )
    write_text(
        directory / "policy.md",
        "# Supplier policy\n\nSuppliers must report Scope 2 emissions annually.\n",
    )
    write_text(
        directory / "memo.txt",
        "The 2021 audit found two minor control weaknesses in procurement.\n",
    )


def _ingest(root: Path, corpus: Path, spec_arguments: list[str]) -> None:
    assert main(["--root", str(root), "ingest", "--source", str(corpus), *spec_arguments]) == 0


def _spec_arguments() -> list[str]:
    return ["--chunk-size", "20", "--overlap", "4"]


def test_search_before_indexing_fails_cleanly(tmp_path: Path, embeddings, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _ingest(root, corpus, _spec_arguments())

    exit_code = main(["--root", str(root), "search", "net zero emissions", "--json"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_index_and_search_round_trip(tmp_path: Path, embeddings, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _ingest(root, corpus, _spec_arguments())

    assert main(["--root", str(root), "index", *()]) == 0
    captured = capsys.readouterr()
    assert "  failure:" not in captured.out

    paths = ProjectPaths(root=root)
    manifest = json.loads((paths.index_dir / "custom-20-4" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vector_count"] > 0
    assert manifest["dimensions"] == DIMENSIONS
    assert manifest["spec"] == "custom-20-4"
    assert manifest["embedding_model"]

    exit_code = main(["--root", str(root), "search", "net zero emissions by 2040", "--json"])
    assert exit_code == 0
    results = json.loads(capsys.readouterr().out)
    assert results["query"] == "net zero emissions by 2040"
    assert results["results"]
    top = results["results"][0]
    assert top["rank"] == 1
    assert top["file_name"] == "report.pdf"
    assert "net zero" in top["text"]
    assert top["page_start"] == 1
    assert 0.0 <= top["distance"] <= 1.0
    assert "citation" in top or "citation_label" in top


def test_reindexing_an_unchanged_corpus_embeds_nothing(tmp_path: Path, embeddings, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _ingest(root, corpus, _spec_arguments())
    assert main(["--root", str(root), "index", "--json"]) == 0
    capsys.readouterr()
    requests_after_first_index = embeddings.request_count

    assert main(["--root", str(root), "index", "--json"]) == 0
    capsys.readouterr()

    assert embeddings.request_count == requests_after_first_index

    index_file = ProjectPaths(root=root).index_dir / "custom-20-4" / "manifest.json"
    manifest = json.loads(index_file.read_text(encoding="utf-8"))
    assert manifest["vector_count"] > 0


def test_index_reports_unknown_spec_as_an_error(tmp_path: Path, embeddings, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _ingest(root, corpus, _spec_arguments())

    exit_code = main(["--root", str(root), "index", "--spec", "does-not-exist"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "does-not-exist" in captured.err


def test_rebuild_replaces_the_index(tmp_path: Path, embeddings, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    _ingest(root, corpus, _spec_arguments())
    assert main(["--root", str(root), "index", "--json"]) == 0
    capsys.readouterr()
    requests_after_first_index = embeddings.request_count

    assert main(["--root", str(root), "index", "--rebuild", "--json"]) == 0
    capsys.readouterr()

    assert embeddings.request_count > requests_after_first_index

    paths = ProjectPaths(root=root)
    manifest = json.loads((paths.index_dir / "custom-20-4" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["vector_count"] > 0

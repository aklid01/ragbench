"""End to end: a corpus is indexed, then ragbench ask grounds and cites an answer.

The test drives the real command line entry point against fake
OpenAI-compatible embedding and LLM servers on localhost, so configuration
loading, retrieval, prompt construction, completion parsing, citation
validation and abstention are all exercised together. The cited-answer run
writes its full JSON payload to a file so the result can be checked and
reproduced outside the test.
"""

from __future__ import annotations

import io
import json
import socket
from pathlib import Path

import pytest

from conftest import write_pdf, write_text
from fake_embeddings import FakeEmbeddingServer
from fake_llm import FakeLLMServer
from ragbench.cli import main
from ragbench.generation import ABSTENTION_MARKER
from ragbench.paths import ProjectPaths

EMBEDDING_KEY = "test-embedding-key"
LLM_KEY = "test-llm-key"


@pytest.fixture
def embeddings(monkeypatch: pytest.MonkeyPatch):
    """A fake embeddings server wired into the process environment."""
    server = FakeEmbeddingServer()
    monkeypatch.setenv("RAGBENCH_EMBEDDING_BASE_URL", server.base_url)
    monkeypatch.setenv("RAGBENCH_EMBEDDING_API_KEY", EMBEDDING_KEY)
    monkeypatch.setenv("RAGBENCH_EMBEDDING_DIMENSIONS", "64")
    monkeypatch.setenv("RAGBENCH_EMBEDDING_BATCH_SIZE", "2")
    monkeypatch.setenv("RAGBENCH_EMBEDDING_MAX_RETRIES", "1")
    monkeypatch.setenv("RAGBENCH_REQUESTS_PER_MINUTE", "100000")
    yield server
    server.stop()


def _wire_llm(monkeypatch: pytest.MonkeyPatch, server: FakeLLMServer) -> FakeLLMServer:
    monkeypatch.setenv("RAGBENCH_LLM_BASE_URL", server.base_url)
    monkeypatch.setenv("RAGBENCH_LLM_API_KEY", LLM_KEY)
    monkeypatch.setenv("RAGBENCH_LLM_MAX_RETRIES", "1")
    return server


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    return root


def _build_corpus(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_pdf(
        directory / "report.pdf",
        [
            "The company commits to reaching net zero emissions across all of its "
            "operations by the year 2040, ten years ahead of the industry average target.",
            "Water consumption across the manufacturing sites fell by 18 percent in 2021 "
            "compared with the previous year, driven by the new recycling program.",
            "The board of directors comprises seven independent members who hold no "
            "financial ties to the company executive leadership team.",
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


def _prepare(root: Path, tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    assert (
        main(
            [
                "--root",
                str(root),
                "ingest",
                "--source",
                str(corpus),
                "--chunk-size",
                "20",
                "--overlap",
                "4",
            ]
        )
        == 0
    )
    assert main(["--root", str(root), "index", "--json"]) == 0
    capsys.readouterr()


def test_ask_returns_a_cited_answer_with_provenance(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    llm = _wire_llm(
        monkeypatch, FakeLLMServer("The company targets net zero emissions by 2040 [1].")
    )
    try:
        exit_code = main(
            ["--root", str(root), "ask", "What is the net zero target year?", "--json"]
        )
    finally:
        llm.stop()
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["question"] == "What is the net zero target year?"
    assert payload["abstained"] is False
    assert "2040" in payload["answer"]
    assert len(payload["retrieved"]) == 5
    assert len(payload["citations"]) == 1
    citation = payload["citations"][0]
    assert citation["reference"] == 1
    assert citation["page_start"] == 1
    assert citation["citation_label"] == "report.pdf (page 1)"
    assert "net zero" in citation["excerpt"].lower()
    assert "2040" in citation["claims"][0]
    cited_chunk = next(
        chunk for chunk in payload["retrieved"] if chunk["chunk_id"] == citation["chunk_id"]
    )
    assert cited_chunk["rank"] == 1
    assert payload["model"] == llm.received[0]["model"]
    assert payload["prompt_tokens"] == 321
    assert payload["completion_tokens"] == 54
    assert payload["total_tokens"] == 375
    assert payload["retrieval_seconds"] >= 0.0
    assert payload["generation_seconds"] >= 0.0

    user_message = llm.received[0]["messages"][1]["content"]
    assert user_message.startswith("Question: What is the net zero target year?")
    assert "[1] report.pdf (page 1)" in user_message
    system_message = llm.received[0]["messages"][0]["content"]
    assert ABSTENTION_MARKER in system_message

    artifact = tmp_path / "artifacts-check" / "answer.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    restored = json.loads(artifact.read_text(encoding="utf-8"))
    assert restored["question"] == payload["question"]
    assert restored["citations"] == payload["citations"]


def test_ask_abstains_when_the_passages_do_not_hold_the_answer(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    llm = _wire_llm(monkeypatch, FakeLLMServer(ABSTENTION_MARKER))
    try:
        exit_code = main(["--root", str(root), "ask", "What is the CEO's golf handicap?", "--json"])
    finally:
        llm.stop()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["abstained"] is True
    assert payload["citations"] == []
    assert payload["invalid_references"] == []


def test_ask_flags_references_to_passages_never_retrieved(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    llm = _wire_llm(monkeypatch, FakeLLMServer("The answer is 2040 [9], confirmed [2]."))
    try:
        exit_code = main(
            ["--root", str(root), "ask", "What is the net zero target year?", "--json"]
        )
    finally:
        llm.stop()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["abstained"] is False
    assert payload["invalid_references"] == [9]
    assert [citation["reference"] for citation in payload["citations"]] == [2]


def test_ask_without_an_index_fails_cleanly(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    assert (
        main(
            [
                "--root",
                str(root),
                "ingest",
                "--source",
                str(corpus),
                "--chunk-size",
                "20",
                "--overlap",
                "4",
            ]
        )
        == 0
    )
    _wire_llm(monkeypatch, FakeLLMServer("unused"))

    exit_code = main(["--root", str(root), "ask", "What is the net zero target year?"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "ragbench index" in captured.err


def test_ask_reports_llm_failures_without_leaking_secrets(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    monkeypatch.setenv("RAGBENCH_LLM_BASE_URL", f"http://127.0.0.1:{dead_port}/v1")
    monkeypatch.setenv("RAGBENCH_LLM_API_KEY", LLM_KEY)
    monkeypatch.setenv("RAGBENCH_LLM_MAX_RETRIES", "0")

    exit_code = main(["--root", str(root), "ask", "What is the net zero target year?"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert LLM_KEY not in captured.err
    assert LLM_KEY not in captured.out


def test_ask_reads_questions_interactively(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    llm = _wire_llm(
        monkeypatch,
        FakeLLMServer(
            [
                "The company targets net zero emissions by 2040 [1].",
                "Water consumption fell by 18 percent in 2021 [1].",
            ]
        ),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("What is the net zero target year?\n"))
    try:
        exit_code = main(["--root", str(root), "ask"])
    finally:
        llm.stop()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.count("question:") == 1
    assert "[1] report.pdf (page 1)" in output
    assert "2040 [1]" in output
    assert "invalid references" not in output
    assert LLM_KEY not in output


def test_ask_accepts_top_k_and_spec_overrides(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    llm = _wire_llm(monkeypatch, FakeLLMServer("The company targets net zero by 2040 [1]."))
    try:
        exit_code = main(
            [
                "--root",
                str(root),
                "ask",
                "What is the net zero target year?",
                "--spec",
                "custom-20-4",
                "--top-k",
                "3",
                "--json",
            ]
        )
    finally:
        llm.stop()

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spec"] == "custom-20-4"
    assert payload["top_k"] == 3
    assert len(payload["retrieved"]) == 3
    index_manifest = json.loads(
        (ProjectPaths(root=root).index_dir / "custom-20-4" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert index_manifest["vector_count"] > 0

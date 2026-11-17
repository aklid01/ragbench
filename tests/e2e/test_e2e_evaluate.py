"""End to end: evaluation and experiment commands over fake servers.

The dataset, corpus, embedding server and LLM server are all fake, so the
full evaluate/experiment/compare path runs without any hosted API. The
numbers assert pipeline wiring, not model quality.
"""

from __future__ import annotations

import json
from datetime import datetime as real_datetime
from pathlib import Path

import pytest

import ragbench.evaluation.runner as runner_module
from conftest import write_pdf
from fake_embeddings import FakeEmbeddingServer
from fake_llm import FakeLLMServer
from ragbench.cli import main

EMBEDDING_KEY = "test-embedding-key"
LLM_KEY = "test-llm-key"

ANSWER = "The company targets net zero emissions by 2040 [1]."


@pytest.fixture
def embeddings(monkeypatch: pytest.MonkeyPatch):
    server = FakeEmbeddingServer()
    monkeypatch.setenv("RAGBENCH_EMBEDDING_BASE_URL", server.base_url)
    monkeypatch.setenv("RAGBENCH_EMBEDDING_API_KEY", EMBEDDING_KEY)
    monkeypatch.setenv("RAGBENCH_EMBEDDING_DIMENSIONS", "64")
    monkeypatch.setenv("RAGBENCH_EMBEDDING_BATCH_SIZE", "8")
    monkeypatch.setenv("RAGBENCH_EMBEDDING_MAX_RETRIES", "1")
    monkeypatch.setenv("RAGBENCH_REQUESTS_PER_MINUTE", "100000")
    yield server
    server.stop()


def _wire_llm(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> FakeLLMServer:
    server = FakeLLMServer(replies)
    monkeypatch.setenv("RAGBENCH_LLM_BASE_URL", server.base_url)
    monkeypatch.setenv("RAGBENCH_LLM_API_KEY", LLM_KEY)
    monkeypatch.setenv("RAGBENCH_LLM_MAX_RETRIES", "1")
    return server


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    return root


def _dataset(root: Path, tmp_path: Path) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "cases": [
                    {
                        "id": "q01",
                        "question": "What is the net zero target year?",
                        "expected_answer": "The company commits to net zero by 2040",
                        "expected_doc": "report.pdf",
                        "expected_text": "reaching net zero emissions across all of its operations by the year 2040",
                        "answerable": True,
                        "notes": "page 1 of report.pdf",
                    },
                    {
                        "id": "q02",
                        "question": "What is the CEO's golf handicap?",
                        "answerable": False,
                        "notes": "not in the corpus",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _prepare(root: Path, tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    write_pdf(
        corpus / "report.pdf",
        [
            "The company commits to reaching net zero emissions across all of its "
            "operations by the year 2040, ten years ahead of the industry average target.",
            "Water consumption across the manufacturing sites fell by 18 percent in 2021 "
            "compared with the previous year, driven by the new recycling program.",
        ],
    )
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


def test_evaluate_runs_the_dataset_and_saves_a_run(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    dataset = _dataset(root, tmp_path)
    llm = _wire_llm(monkeypatch, [ANSWER, "INSUFFICIENT_CONTEXT"])
    try:
        exit_code = main(
            ["--root", str(root), "evaluate", "--dataset", str(dataset), "--top-k", "3", "--json"]
        )
    finally:
        llm.stop()
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["spec"].startswith("custom-20-4")
    assert payload["config"]["top_k"] == 3
    assert payload["config"]["dataset_version"] == "1.0.0"
    assert payload["operational"]["total_cases"] == 2
    assert payload["operational"]["errors"] == 0
    # One answerable case retrieved and answered, one unanswerable abstained.
    assert payload["generation"]["abstention_rate_unanswerable"] == 1.0
    assert payload["generation"]["answered_answerable_rate"] == 1.0
    assert payload["generation"]["citation_precision"] == 1.0
    assert payload["retrieval"]["recall"]["3"] == 1.0

    run_file = Path(payload["run_file"])
    assert run_file.is_file()
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    assert len(saved["records"]) == 2
    answerable = next(record for record in saved["records"] if record["case_id"] == "q01")
    assert answerable["rank_of_expected"] == 1
    assert answerable["cited_expected"] is True
    assert answerable["answer"].startswith("The company targets")
    unanswerable = next(record for record in saved["records"] if record["case_id"] == "q02")
    assert unanswerable["abstained"] is True


def test_evaluate_reports_llm_failure_as_case_error(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    dataset = _dataset(root, tmp_path)
    llm = _wire_llm(monkeypatch, [])
    llm.stop()  # dead endpoint: every case must record an error, not crash

    exit_code = main(
        ["--root", str(root), "evaluate", "--dataset", str(dataset), "--top-k", "3", "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operational"]["errors"] == 2
    assert all(record["error"] for record in payload["errors"]) or payload["errors"]
    # Every case errored, so retrieval metrics are not computable, not zero.
    assert payload["retrieval"]["recall"]["3"] is None
    assert payload["retrieval"]["mrr"]["3"] is None


def test_experiment_runs_a_grid_and_compare_prints_the_table(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    dataset = _dataset(root, tmp_path)
    llm = _wire_llm(monkeypatch, [ANSWER, "INSUFFICIENT_CONTEXT"])
    try:
        exit_code = main(
            [
                "--root",
                str(root),
                "experiment",
                "--dataset",
                str(dataset),
                "--specs",
                "custom-20-4",
                "--top-ks",
                "3,5",
                "--json",
            ]
        )
    finally:
        llm.stop()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "2 runs saved" in output

    exit_code = main(["--root", str(root), "compare"])
    assert exit_code == 0
    table = capsys.readouterr().out
    assert "recall@3" in table
    assert table.count("\n") >= 3


def test_experiment_dry_run_needs_no_apis(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    exit_code = main(
        [
            "--root",
            str(root),
            "experiment",
            "--specs",
            "small,medium",
            "--top-ks",
            "3,5",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[0] == "small-k3 (chunking=small, top_k=3)"
    assert lines[-1] == "medium-k5 (chunking=medium, top_k=5)"


def test_evaluate_without_a_dataset_fails_cleanly(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    exit_code = main(["--root", str(root), "evaluate"])
    assert exit_code == 1
    assert "no evaluation dataset" in capsys.readouterr().err


def test_two_runs_in_one_second_do_not_overwrite(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    dataset = _dataset(root, tmp_path)
    llm = _wire_llm(monkeypatch, [ANSWER, "INSUFFICIENT_CONTEXT"])

    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 5, 1, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(runner_module, "datetime", FrozenDatetime)
    try:
        for _ in range(2):
            assert (
                main(
                    [
                        "--root",
                        str(root),
                        "evaluate",
                        "--dataset",
                        str(dataset),
                        "--top-k",
                        "3",
                        "--json",
                    ]
                )
                == 0
            )
            capsys.readouterr()
    finally:
        llm.stop()

    run_files = sorted((root / "artifacts" / "experiments" / "runs").glob("*.json"))
    ids = {json.loads(path.read_text(encoding="utf-8"))["summary"]["run_id"] for path in run_files}
    assert ids == {"20260501-120000-custom-20-4-k3", "20260501-120000-custom-20-4-k3-2"}


def test_experiment_skips_a_leftover_index_directory(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    dataset = _dataset(root, tmp_path)
    (root / "artifacts" / "index" / "ghost-spec").mkdir(parents=True, exist_ok=True)
    llm = _wire_llm(monkeypatch, [ANSWER, "INSUFFICIENT_CONTEXT"])
    try:
        exit_code = main(
            [
                "--root",
                str(root),
                "experiment",
                "--dataset",
                str(dataset),
                "--specs",
                "custom-20-4,ghost-spec",
                "--top-ks",
                "3,5",
            ]
        )
    finally:
        llm.stop()
    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.count("skipping") == 2
    assert "2 runs saved" in output


def test_abstention_is_detected_when_the_model_appends_prose(
    tmp_path: Path, embeddings, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    root = _make_project(tmp_path)
    _prepare(root, tmp_path, capsys)
    dataset = _dataset(root, tmp_path)
    llm = _wire_llm(
        monkeypatch,
        [
            "INSUFFICIENT_CONTEXT.\nThe passages do not mention a target year.",
            "The company targets 2040 [1]. The context is not INSUFFICIENT_CONTEXT.",
        ],
    )
    try:
        exit_code = main(
            ["--root", str(root), "evaluate", "--dataset", str(dataset), "--top-k", "3", "--json"]
        )
    finally:
        llm.stop()
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    saved = json.loads(Path(payload["run_file"]).read_text(encoding="utf-8"))
    answerable = next(record for record in saved["records"] if record["case_id"] == "q01")
    unanswerable = next(record for record in saved["records"] if record["case_id"] == "q02")
    assert answerable["abstained"] is True
    assert answerable["citations"] == []
    assert unanswerable["abstained"] is False
    assert unanswerable["citations"]

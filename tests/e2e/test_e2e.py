"""End to end: a corpus goes in, a queryable set of artefacts comes out.

The test drives the real command line entry point, so configuration
resolution, ingestion, chunking and persistence are all exercised together.
It writes its artefacts into a temporary project and checks them afterwards,
which is the artefact the run can be reproduced from.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import write_pdf, write_text
from ragbench.cli import main
from ragbench.paths import ProjectPaths


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


def test_an_unreadable_document_fails_the_command_without_stopping_the_run(
    tmp_path: Path, capsys
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "broken.pdf").write_bytes(b"this is not a pdf file")
    write_text(corpus / "memo.txt", "The 2021 audit found two minor control weaknesses.\n")

    exit_code = main(
        [
            "--root",
            str(root),
            "ingest",
            "--source",
            str(corpus),
            "--no-chunk",
        ]
    )
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "failure: broken.pdf:" in output
    assert "ingested: 1 documents" in output
    assert "1 failures" in output


def test_ingest_produces_a_complete_artefact_set(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)

    exit_code = main(
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
    assert exit_code == 0

    paths = ProjectPaths(root=root)
    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == 3
    assert {entry["file_name"] for entry in manifest["documents"]} == {
        "report.pdf",
        "policy.md",
        "memo.txt",
    }

    pdf_entry = next(e for e in manifest["documents"] if e["file_name"] == "report.pdf")
    assert pdf_entry["page_count"] == 3
    assert pdf_entry["extractor"] == "pymupdf"

    chunk_files = sorted((paths.chunks_dir / "custom-20-4").glob("*.json"))
    assert len(chunk_files) == 3
    total_chunks = 0
    for chunk_file in chunk_files:
        payload = json.loads(chunk_file.read_text(encoding="utf-8"))
        assert payload["chunk_count"] == len(payload["chunks"])
        assert payload["chunks"]
        total_chunks += payload["chunk_count"]
        for chunk in payload["chunks"]:
            assert chunk["text"].strip()
            assert chunk["token_count"] <= 20
            assert chunk["chunk_id"].startswith(payload["doc_id"])
    assert total_chunks > 3

    captured = capsys.readouterr()
    assert "3 documents" in captured.out
    assert "  failure:" not in captured.out


def test_the_ingested_artefacts_answer_where_text_came_from(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
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

    paths = ProjectPaths(root=root)
    payload = json.loads(
        (paths.chunks_dir / "custom-20-4" / f"{_pdf_doc_id(paths)}.json").read_text(
            encoding="utf-8"
        )
    )
    joined = " ".join(chunk["text"] for chunk in payload["chunks"])
    assert "net zero emissions by 2040" in joined
    assert "Water consumption fell by 18 percent" in joined
    assert "seven independent directors" in joined
    starts = {chunk["page_start"] for chunk in payload["chunks"]}
    assert starts <= {1, 2, 3}
    assert payload["chunks"][-1]["page_end"] == 3


def test_reingesting_an_edited_corpus_retires_the_old_document(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    arguments = [
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
    assert main(arguments) == 0

    paths = ProjectPaths(root=root)
    old_memo_id = _doc_id_for_file(paths, "memo.txt")
    write_text(
        corpus / "memo.txt",
        "The 2022 audit found three major control weaknesses in procurement.\n",
    )

    assert main(arguments) == 0
    output = capsys.readouterr().out
    assert "retired:" in output
    assert old_memo_id in output

    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["documents"]) == 3
    assert old_memo_id not in {entry["doc_id"] for entry in manifest["documents"]}
    assert not (paths.extracted_dir / f"{old_memo_id}.json").exists()
    assert not (paths.chunks_dir / "custom-20-4" / f"{old_memo_id}.json").exists()
    new_memo_id = _doc_id_for_file(paths, "memo.txt")
    assert new_memo_id != old_memo_id
    assert (paths.chunks_dir / "custom-20-4" / f"{new_memo_id}.json").is_file()


def test_manifest_describes_the_whole_corpus_after_two_ingests(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus_a = tmp_path / "corpus_a"
    corpus_b = tmp_path / "corpus_b"
    corpus_a.mkdir()
    corpus_b.mkdir()
    write_text(corpus_a / "memo.txt", "The 2021 audit found two minor control weaknesses.\n")
    write_text(corpus_b / "policy.md", "# Supplier policy\n\nSuppliers must report Scope 2.\n")

    assert (
        main(
            [
                "--root",
                str(root),
                "ingest",
                "--source",
                str(corpus_a),
                "--chunk-size",
                "20",
                "--overlap",
                "4",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--root",
                str(root),
                "ingest",
                "--source",
                str(corpus_b),
                "--chunk-size",
                "20",
                "--overlap",
                "4",
            ]
        )
        == 0
    )

    paths = ProjectPaths(root=root)
    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == 2
    assert {entry["file_name"] for entry in manifest["documents"]} == {"memo.txt", "policy.md"}
    listed = sorted(path.stem for path in paths.extracted_dir.glob("*.json"))
    assert listed == ["manifest", *sorted(entry["doc_id"] for entry in manifest["documents"])]


def test_running_the_pipeline_twice_changes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)
    arguments = [
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
    assert main(arguments) == 0
    first = _chunk_content(ProjectPaths(root=root))
    assert main(arguments) == 0
    second = _chunk_content(ProjectPaths(root=root))
    assert first == second


def test_inspect_commands_report_the_state(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
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
    capsys.readouterr()

    assert main(["--root", str(root), "inspect", "documents"]) == 0
    listed = capsys.readouterr().out
    assert "report.pdf" in listed

    assert (
        main(
            [
                "--root",
                str(root),
                "inspect",
                "chunks",
                "--doc-id",
                _pdf_doc_id(ProjectPaths(root=root)),
            ]
        )
        == 0
    )
    inspected = capsys.readouterr().out
    assert "#c0000" in inspected
    assert "page" in inspected


def _pdf_doc_id(paths: ProjectPaths) -> str:
    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["documents"]:
        if entry["file_name"] == "report.pdf":
            return entry["doc_id"]
    raise AssertionError("report.pdf was not ingested")


def _doc_id_for_file(paths: ProjectPaths, file_name: str) -> str:
    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["documents"]:
        if entry["file_name"] == file_name:
            return entry["doc_id"]
    raise AssertionError(f"{file_name} was not ingested")


def _chunk_content(paths: ProjectPaths) -> dict[str, list[dict[str, object]]]:
    """The chunk payloads without the ingestion timestamp, which changes per run."""
    content: dict[str, list[dict[str, object]]] = {}
    for path in sorted((paths.chunks_dir / "custom-20-4").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for chunk in payload["chunks"]:
            chunk["source"] = {
                key: value for key, value in chunk["source"].items() if key != "ingested_at"
            }
        content[path.name] = payload["chunks"]
    return content


def test_ingest_json_payload_reports_both_summaries(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)

    exit_code = main(
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
            "--json",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"ingest", "chunking"}
    ingest = payload["ingest"]
    assert len(ingest["documents"]) == 3
    assert ingest["pages"] == 5
    assert ingest["characters"] > 0
    assert ingest["duplicates"] == []
    assert ingest["failures"] == []
    assert ingest["unchanged"] == []
    assert ingest["retired"] == []
    chunking = payload["chunking"]
    assert chunking is not None
    assert chunking["spec"] == "custom-20-4"
    assert chunking["documents"] == 3
    assert chunking["chunks"] > 3
    assert chunking["tokens"] > 0
    assert chunking["largest_chunk_tokens"] <= 20
    assert chunking["failures"] == []


def test_ingest_json_without_chunking_reports_a_null_chunking_section(
    tmp_path: Path, capsys
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)

    assert (
        main(["--root", str(root), "ingest", "--source", str(corpus), "--no-chunk", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["ingest"]["documents"]) == 3
    assert payload["chunking"] is None


def test_ingest_force_preset_and_unchanged_paths(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    corpus = tmp_path / "corpus"
    _build_corpus(corpus)

    preset_arguments = [
        "--root",
        str(root),
        "ingest",
        "--source",
        str(corpus),
        "--preset",
        "small",
    ]
    assert main(preset_arguments) == 0
    paths = ProjectPaths(root=root)
    assert (paths.chunks_dir / "small-500").is_dir()

    # A plain re-run over an unchanged corpus skips every file.
    capsys.readouterr()
    assert main(preset_arguments) == 0
    output = capsys.readouterr().out
    assert "3 already extracted, skipped" in output

    # A content duplicate (different bytes, same normalised text) is skipped
    # on a plain re-run; the summary names the file it duplicates.
    (corpus / "memo_copy.md").write_bytes(
        b"The 2021 audit found two minor control weaknesses in procurement.\r\n\r\n"
    )
    capsys.readouterr()
    assert main(preset_arguments) == 0
    output = capsys.readouterr().out
    assert "duplicate: memo_copy.md holds the same text as memo.txt" in output
    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == 3

    # --force re-extracts everything and keeps the text duplicate in place.
    assert main([*preset_arguments, "--force"]) == 0
    output = capsys.readouterr().out
    assert "0 duplicates skipped" in output
    manifest = json.loads((paths.extracted_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["document_count"] == 4


def test_config_command_prints_settings_without_secrets(tmp_path: Path, capsys) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")

    assert main(["--root", str(root), "config"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["root"] == str(root)
    assert payload["llm"]["api_key"] == "not set"
    assert payload["embedding"]["api_key"] == "not set"
    assert payload["chunking"]["chunk_size_tokens"] == 800

    assert main(["--root", str(root), "config", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload

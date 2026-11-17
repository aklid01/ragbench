"""Ingestion: extraction, provenance, duplicates and failure handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_pdf, write_text
from ragbench.errors import ExtractionError
from ragbench.ingest.discover import discover_documents
from ragbench.ingest.extract import detect_media_type, extract_document
from ragbench.ingest.runner import run_ingest


def test_pdf_pages_keep_their_page_numbers(tmp_path: Path) -> None:
    path = write_pdf(
        tmp_path / "report.pdf", ["First page about climate.", "Second page about people."]
    )
    result = extract_document(path)
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "First page about climate." in result.pages[0].text
    assert "Second page about people." in result.pages[1].text


def test_empty_pdf_page_keeps_its_number(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "gaps.pdf", ["Page one.", "", "Page three."])
    result = extract_document(path)
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.pages[1].text == ""


def test_text_file_has_no_page_numbers(tmp_path: Path) -> None:
    path = write_text(tmp_path / "notes.txt", "Some plain notes about reporting.")
    result = extract_document(path)
    assert len(result.pages) == 1
    assert result.pages[0].page_number is None
    assert "plain notes" in result.pages[0].text


def test_markdown_is_kept_verbatim(tmp_path: Path) -> None:
    body = "# Heading\n\n- a bullet\n- another bullet\n"
    path = write_text(tmp_path / "readme.md", body)
    result = extract_document(path)
    assert "# Heading" in result.pages[0].text
    assert "- a bullet" in result.pages[0].text


def test_media_type_detection(tmp_path: Path) -> None:
    assert detect_media_type(tmp_path / "a.pdf") == "pdf"
    assert detect_media_type(tmp_path / "a.txt") == "text"
    assert detect_media_type(tmp_path / "a.md") == "markdown"
    assert detect_media_type(tmp_path / "a.docx") is None


def test_broken_pdf_is_reported_as_a_failure(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is definitely not a pdf")
    try:
        extract_document(broken)
    except ExtractionError:
        return
    raise AssertionError("a broken PDF must raise ExtractionError")


def test_byte_duplicates_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_text(source / "one.txt", "The same content.")
    write_text(source / "two.txt", "The same content.")
    result = discover_documents(source)
    assert len(result.files) == 1
    assert len(result.duplicates) == 1
    assert "byte for byte" in result.duplicates[0]


def test_content_duplicates_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_text(source / "one.txt", "The same content.\n")
    (source / "two.md").write_bytes(b"The same content.\r\n\r\n")
    from ragbench.paths import ProjectPaths

    paths = ProjectPaths(root=tmp_path).ensure()
    summary = run_ingest(paths, source)
    assert len(summary.documents) == 1
    assert len(summary.duplicates) == 1
    assert "same text" in summary.duplicates[0]


def test_ingest_records_counts_and_writes_documents(tmp_path: Path, paths) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_pdf(source / "report.pdf", ["Alpha page.", "Beta page."])
    write_text(source / "notes.txt", "Plain notes.")
    write_text(source / "empty.txt", "")

    summary = run_ingest(paths, source)

    assert len(summary.documents) == 3
    assert summary.page_count == 4
    assert summary.failures == []
    assert any("no extractable text" in warning for warning in summary.warnings)
    for doc_id in summary.documents:
        assert (paths.extracted_dir / f"{doc_id}.json").is_file()


def test_ingest_is_idempotent(tmp_path: Path, paths) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_text(source / "notes.txt", "Plain notes about reporting.")
    first = run_ingest(paths, source)
    second = run_ingest(paths, source)
    assert second.unchanged == first.documents
    assert second.documents == []
    assert second.retired == []
    assert second.char_count == 0


def test_unsupported_files_are_ignored(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_text(source / "notes.txt", "Plain notes.")
    (source / "image.png").write_bytes(b"\x89PNG")
    result = discover_documents(source)
    assert [file.path.name for file in result.files] == ["notes.txt"]


def test_multipage_documents_with_the_same_first_page_are_not_duplicates(
    tmp_path: Path, paths
) -> None:
    """Regression: dedup used to hash only page 1, collapsing distinct PDFs."""
    source = tmp_path / "source"
    source.mkdir()
    write_pdf(
        source / "report_a.pdf",
        ["Shared cover page.", "Report A about solar panels."],
    )
    write_pdf(
        source / "report_b.pdf",
        ["Shared cover page.", "Report B about wind turbines."],
    )
    summary = run_ingest(paths, source)
    assert len(summary.documents) == 2
    assert summary.duplicates == []


def test_content_duplicate_detection_covers_the_whole_document(
    tmp_path: Path, paths
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_pdf(
        source / "report_a.pdf",
        ["Shared cover page.", "Report A about solar panels."],
    )
    write_pdf(
        source / "report_b.pdf",
        ["Shared cover page.", "Report A about solar panels."],
    )
    summary = run_ingest(paths, source)
    assert len(summary.documents) == 1
    assert len(summary.duplicates) == 1


def test_encrypted_pdf_reports_a_clear_failure(tmp_path: Path, paths) -> None:
    import pymupdf

    source = tmp_path / "source"
    source.mkdir()
    path = source / "locked.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "secret text")
    document.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    document.close()

    summary = run_ingest(paths, source)
    assert summary.documents == []
    assert len(summary.failures) == 1
    assert "locked.pdf" in summary.failures[0]
    assert "encrypted" in summary.failures[0]


def test_undecodable_bytes_report_a_clear_failure(tmp_path: Path, paths) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "broken.txt").write_bytes(b"fine start \x81\xff never decodes")
    summary = run_ingest(paths, source)
    assert summary.documents == []
    assert len(summary.failures) == 1
    assert "decode" in summary.failures[0]


def test_per_page_read_failure_is_recorded_as_a_warning(
    tmp_path: Path, paths, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page whose text cannot be read degrades to empty, it never fails the run.

    MuPDF itself degrades corrupt streams to empty text, so the failure is
    simulated at the ``Page.get_text`` boundary.
    """
    import pymupdf

    source = tmp_path / "source"
    source.mkdir()
    write_pdf(source / "report.pdf", ["Good page.", "Damaged page.", "Final page."])

    original_get_text = pymupdf.Page.get_text
    calls = {"n": 0}

    def flaky_get_text(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated MuPDF failure")
        return original_get_text(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_text", flaky_get_text)
    summary = run_ingest(paths, source)

    assert len(summary.documents) == 1
    from ragbench.store import load_document

    stored = load_document(paths, summary.documents[0])
    assert [page.text for page in stored.pages] == ["Good page.", "", "Final page."]
    assert any("page 2 could not be read" in warning for warning in stored.warnings)

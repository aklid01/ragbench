"""Shared test fixtures.

The tests never call a hosted API. Documents are built in the test itself,
so the suite has no binary fixtures to keep in step with the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.chunking.tokenizer import get_counter
from ragbench.config import load_settings
from ragbench.paths import ProjectPaths


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """An empty project root."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'ragbench-test'\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def paths(project_root: Path) -> ProjectPaths:
    """The project layout rooted at the temporary project."""
    return ProjectPaths(root=project_root).ensure()


@pytest.fixture
def settings(project_root: Path):
    """Settings that ignore the machine's environment."""
    return load_settings(env={}, root=project_root, use_env_file=False)


@pytest.fixture
def counter():
    """The default tokenizer."""
    return get_counter("cl100k_base")


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Build a PDF with one text line per page."""
    import pymupdf

    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 100), text)
    document.save(path)
    document.close()
    return path


def write_text(path: Path, text: str) -> Path:
    """Write a plain text file."""
    path.write_text(text, encoding="utf-8")
    return path


def make_document(source_file: str = "report.pdf", pages: list[str] | None = None):
    """Build an extracted document without going through the file system."""
    from ragbench.models import ExtractedDocument, Page, SourceRef

    page_list = [
        Page(page_number=index + 1, text=text)
        for index, text in enumerate(pages or ["Page one text.", "Page two text."])
    ]
    return ExtractedDocument(
        doc_id="report-1234abcd",
        title="report",
        source=SourceRef(
            path=f"/tmp/{source_file}",
            file_name=source_file,
            sha256="1234abcd" * 8,
            media_type="pdf",
            byte_size=1024,
            ingested_at="2026-01-01T00:00:00Z",
        ),
        pages=page_list,
        extractor="pymupdf",
    )

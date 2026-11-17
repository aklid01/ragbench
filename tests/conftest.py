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
from ragbench.retrieval.search import RetrievedChunk


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


def make_chunk(rank: int, file_name: str, page_start: int | None, text: str) -> RetrievedChunk:
    """Build one retrieved chunk with the provenance the pipeline would attach."""
    return RetrievedChunk(
        rank=rank,
        chunk_id=f"{file_name}-chunk-{rank}",
        doc_id=f"{file_name}-doc",
        file_name=file_name,
        page_start=page_start,
        page_end=page_start,
        text=text,
        distance=0.2,
        similarity=0.8,
    )


def make_case_record(case_id: str, **overrides):
    """Build a CaseRecord with sane defaults, overridden per test."""
    from ragbench.evaluation.records import CaseRecord

    defaults = {
        "case_id": case_id,
        "question": f"question {case_id}?",
        "answerable": True,
        "expected_doc": "report.pdf",
        "expected_text": "the expected passage text",
    }
    defaults.update(overrides)
    return CaseRecord.model_validate(defaults)


def make_answered_question(
    answer: str,
    retrieved_texts: list[str],
    citation_reference: int | None = None,
):
    """Build an AnsweredQuestion like the generation pipeline would."""
    from datetime import UTC, datetime

    from ragbench.generation.citations import build_citations
    from ragbench.generation.pipeline import AnsweredQuestion
    from ragbench.retrieval.search import RetrievedChunk

    retrieved = [
        RetrievedChunk(
            rank=index + 1,
            chunk_id=f"chunk-{index + 1}",
            doc_id="report-1234abcd",
            file_name="report.pdf",
            page_start=1,
            page_end=1,
            text=text,
            distance=0.3,
            similarity=0.7,
        )
        for index, text in enumerate(retrieved_texts)
    ]
    citations, invalid = build_citations(answer, retrieved) if citation_reference else ([], [])
    return AnsweredQuestion(
        question="What is the net zero year?",
        spec="test-800",
        top_k=5,
        answer=answer,
        abstained=False,
        citations=citations,
        invalid_references=invalid,
        retrieved=retrieved,
        embedding_model="fake-embed",
        model="fake-llm",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        retrieval_seconds=0.01,
        generation_seconds=0.02,
        answered_at=datetime.now(UTC),
    )

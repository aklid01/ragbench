"""Ingestion pipeline: discovery, extraction, duplicate detection, persistence.

A single unreadable file must not stop a run and an empty document must not
produce a silent success; both end up in the summary. Files already extracted
under the same id are skipped unless ``force`` re-extracts them, documents of
the ingested source directory whose file changed or vanished are retired, and
the manifest always describes the whole extracted corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..errors import ExtractionError, IngestionError
from ..models import ExtractedDocument, SourceRef, content_hash, doc_id_for
from ..paths import ProjectPaths
from ..store import MANIFEST_NAME, write_document, write_manifest
from .discover import DiscoveredFile, discover_documents
from .extract import ExtractionResult, extract_document

__all__ = ["IngestSummary", "run_ingest"]


@dataclass
class IngestSummary:
    """What an ingest run did."""

    documents: list[str] = field(default_factory=list)
    page_count: int = 0
    char_count: int = 0
    duplicates: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def describe(self) -> str:
        return (
            f"{len(self.documents)} documents, {self.page_count} pages, "
            f"{self.char_count} characters, {len(self.duplicates)} duplicates skipped, "
            f"{len(self.failures)} failures"
        )


def run_ingest(
    paths: ProjectPaths,
    source_dir: Path,
    *,
    force: bool = False,
) -> IngestSummary:
    """Extract every supported document below ``source_dir`` into ``paths``.

    Files whose id is already extracted are skipped unless ``force``
    re-extracts them (keeping text duplicates). Afterwards the documents of
    this source directory whose file changed or disappeared are retired, and
    the manifest is rewritten from the whole extracted corpus.
    """
    summary = IngestSummary()
    paths.ensure()
    source_dir = Path(source_dir).resolve()

    try:
        discovery = discover_documents(source_dir)
    except FileNotFoundError as exc:
        raise IngestionError(str(exc)) from exc

    summary.duplicates.extend(discovery.duplicates)
    summary.skipped.extend(discovery.skipped)
    discovered_by_path = {file.path: file.sha256 for file in discovery.files}

    seen_text_hashes: dict[str, str] = {}

    for discovered in discovery.files:
        doc_id = doc_id_for(discovered.path.name, discovered.sha256)
        existing = None if force else _load_extracted(paths, doc_id)
        if existing is not None:
            _reuse_extracted(existing, discovered, seen_text_hashes, summary)
            continue
        extraction = _extract_discovered(discovered, summary)
        if extraction is None:
            continue
        _persist_extracted(
            paths,
            discovered,
            extraction,
            doc_id,
            force=force,
            seen=seen_text_hashes,
            summary=summary,
        )

    _reconcile_corpus(paths, source_dir, discovered_by_path, summary)
    return summary


def _reuse_extracted(
    existing: ExtractedDocument,
    discovered: DiscoveredFile,
    seen_text_hashes: dict[str, str],
    summary: IngestSummary,
) -> None:
    """Record an already extracted file as unchanged or as a text duplicate."""
    doc_id = doc_id_for(discovered.path.name, discovered.sha256)
    text_hash = content_hash(existing.pages[0].text if existing.pages else "")
    if text_hash in seen_text_hashes:
        summary.duplicates.append(
            f"{discovered.path.name} holds the same text as {seen_text_hashes[text_hash]}"
        )
        return
    seen_text_hashes.setdefault(text_hash, discovered.path.name)
    summary.unchanged.append(doc_id)


def _extract_discovered(
    discovered: DiscoveredFile, summary: IngestSummary
) -> ExtractionResult | None:
    """Extract one file; a failure is recorded and yields None, never an exception."""
    try:
        return extract_document(discovered.path)
    except ExtractionError as exc:
        summary.failures.append(f"{discovered.path.name}: {exc}")
    except Exception as exc:
        summary.failures.append(f"{discovered.path.name}: unexpected error: {exc}")
    return None


def _persist_extracted(
    paths: ProjectPaths,
    discovered: DiscoveredFile,
    extraction: ExtractionResult,
    doc_id: str,
    *,
    force: bool,
    seen: dict[str, str],
    summary: IngestSummary,
) -> None:
    """Dedup, build and write the extracted document, updating the summary."""
    text_hash = content_hash(extraction.pages[0].text if extraction.pages else "")
    if text_hash in seen and not force:
        summary.duplicates.append(
            f"{discovered.path.name} holds the same text as {seen[text_hash]}"
        )
        return
    seen.setdefault(text_hash, discovered.path.name)

    document = ExtractedDocument(
        doc_id=doc_id,
        title=extraction.title or discovered.path.stem,
        source=SourceRef(
            path=str(discovered.path),
            file_name=discovered.path.name,
            sha256=discovered.sha256,
            media_type=discovered.media_type,
            byte_size=discovered.byte_size,
            ingested_at=datetime.now(UTC),
        ),
        pages=extraction.pages,
        extractor=extraction.extractor,
        warnings=extraction.warnings,
    )

    if document.char_count == 0:
        summary.warnings.append(f"{document.source.file_name} contains no extractable text")

    write_document(paths, document)
    summary.documents.append(document.doc_id)
    summary.page_count += document.page_count
    summary.char_count += document.char_count
    summary.warnings.extend(document.warnings)


def _load_extracted(paths: ProjectPaths, doc_id: str) -> ExtractedDocument | None:
    """The stored document, or None when it is missing or unreadable."""
    source = paths.extracted_dir / f"{doc_id}.json"
    if not source.is_file():
        return None
    try:
        return ExtractedDocument.model_validate_json(source.read_text(encoding="utf-8"))
    except ValidationError:
        return None


def _reconcile_corpus(
    paths: ProjectPaths,
    source_dir: Path,
    discovered_by_path: dict[Path, str],
    summary: IngestSummary,
) -> None:
    """Retire stale documents of this source dir and write the corpus manifest.

    A stored document is retired when its source file sits below the ingested
    source directory but is gone now, or holds different content now. Only
    this source directory is reconciled; documents ingested from other
    directories are left alone. The manifest is then rewritten from whatever
    remains, so it describes the corpus and not the last batch.
    """
    kept: list[ExtractedDocument] = []
    for path in sorted(paths.extracted_dir.glob("*.json")):
        if path.name == MANIFEST_NAME:
            continue
        try:
            document = ExtractedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            summary.warnings.append(f"unreadable extracted document left in place: {path.name}")
            continue
        doc_path = Path(document.source.path)
        try:
            under_source = doc_path.resolve().is_relative_to(source_dir)
            current_sha = discovered_by_path.get(doc_path.resolve())
        except OSError:
            under_source, current_sha = False, None
        if under_source and current_sha != document.source.sha256:
            path.unlink()
            summary.retired.append(f"{document.source.file_name} ({document.doc_id})")
            continue
        kept.append(document)
    write_manifest(paths, kept)

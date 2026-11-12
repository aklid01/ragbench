"""Ingestion pipeline.

Discovery, extraction, duplicate detection and persistence, in that order.
The pipeline is defensive by design: a single unreadable file must not stop
a run, and an empty document must not produce a silent success. Both end up
in the summary, which is the only place the caller has to look to learn what
actually happened.

Ingestion is idempotent. Running it twice over the same directory produces
the same document ids and the same files, because the ids are derived from
the file content rather than from the clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..errors import ExtractionError, IngestionError
from ..models import ExtractedDocument, SourceRef, content_hash, doc_id_for
from ..paths import ProjectPaths
from ..store import write_document, write_manifest
from .discover import discover_documents
from .extract import extract_document

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

    ``force`` re-extracts documents that were already ingested. Without it a
    document whose file content has not changed is left alone.
    """
    summary = IngestSummary()
    paths.ensure()

    try:
        discovery = discover_documents(source_dir)
    except FileNotFoundError as exc:
        raise IngestionError(str(exc)) from exc

    summary.duplicates.extend(discovery.duplicates)
    summary.skipped.extend(discovery.skipped)

    documents: list[ExtractedDocument] = []
    seen_text_hashes: dict[str, str] = {}

    for discovered in discovery.files:
        try:
            extraction = extract_document(discovered.path)
        except ExtractionError as exc:
            summary.failures.append(f"{discovered.path.name}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            summary.failures.append(f"{discovered.path.name}: unexpected error: {exc}")
            continue

        text_hash = content_hash(extraction.pages[0].text if extraction.pages else "")
        if text_hash in seen_text_hashes and not force:
            summary.duplicates.append(
                f"{discovered.path.name} holds the same text as {seen_text_hashes[text_hash]}"
            )
            continue
        seen_text_hashes.setdefault(text_hash, discovered.path.name)

        document = ExtractedDocument(
            doc_id=doc_id_for(discovered.path.name, discovered.sha256),
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
        documents.append(document)
        summary.documents.append(document.doc_id)
        summary.page_count += document.page_count
        summary.char_count += document.char_count
        summary.warnings.extend(document.warnings)

    write_manifest(paths, documents)
    return summary

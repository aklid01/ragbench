"""Persistence of extracted documents, chunk files and the ingest manifest.

All reads and writes of the workbench's own artefacts go through this module,
so the on-disk layout is described in exactly one place.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from .errors import ArtifactError
from .models import Chunk, ExtractedDocument
from .paths import ProjectPaths

__all__ = [
    "MANIFEST_NAME",
    "ManifestEntry",
    "IngestManifest",
    "write_json",
    "write_document",
    "load_document",
    "list_documents",
    "write_manifest",
    "read_manifest",
    "write_chunks",
    "load_chunks",
]

MANIFEST_NAME = "manifest.json"


class ManifestEntry(TypedDict):
    """One document as recorded in the ingest manifest."""

    doc_id: str
    file_name: str
    path: str
    sha256: str
    media_type: str
    page_count: int
    char_count: int
    extractor: str
    ingested_at: str
    warnings: list[str]


class IngestManifest(TypedDict):
    """The shape of the ingest manifest file."""

    generated_at: str | None
    document_count: int
    documents: list[ManifestEntry]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temp file and rename, so a crash cannot corrupt the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_document(paths: ProjectPaths, document: ExtractedDocument) -> Path:
    """Write one extracted document and return the file it landed in."""
    target = paths.extracted_dir / f"{document.doc_id}.json"
    write_json(target, document.model_dump(mode="json"))
    return target


def load_document(paths: ProjectPaths, doc_id: str) -> ExtractedDocument:
    """Read one extracted document back."""
    source = paths.extracted_dir / f"{doc_id}.json"
    if not source.is_file():
        raise ArtifactError(f"no extracted document with id {doc_id!r}")
    return ExtractedDocument.model_validate_json(source.read_text(encoding="utf-8"))


def list_documents(paths: ProjectPaths) -> list[str]:
    """Ids of every extracted document, sorted."""
    if not paths.extracted_dir.is_dir():
        return []
    return sorted(
        path.stem for path in paths.extracted_dir.glob("*.json") if path.stem != "manifest"
    )


def write_manifest(paths: ProjectPaths, documents: list[ExtractedDocument]) -> Path:
    """Write the ingest manifest."""
    payload: IngestManifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "document_count": len(documents),
        "documents": [
            {
                "doc_id": document.doc_id,
                "file_name": document.source.file_name,
                "path": document.source.path,
                "sha256": document.source.sha256,
                "media_type": document.source.media_type,
                "page_count": document.page_count,
                "char_count": document.char_count,
                "extractor": document.extractor,
                "ingested_at": document.source.ingested_at.isoformat(),
                "warnings": document.warnings,
            }
            for document in documents
        ],
    }
    target = paths.extracted_dir / MANIFEST_NAME
    write_json(target, dict(payload))
    return target


def read_manifest(paths: ProjectPaths) -> IngestManifest:
    """Read the ingest manifest, or an empty one when it does not exist yet."""
    source = paths.extracted_dir / MANIFEST_NAME
    if not source.is_file():
        return {"generated_at": None, "document_count": 0, "documents": []}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {
        "generated_at": payload.get("generated_at"),
        "document_count": payload.get("document_count", 0),
        "documents": payload.get("documents", []),
    }


def write_chunks(paths: ProjectPaths, spec_name: str, doc_id: str, chunks: list[Chunk]) -> Path:
    """Write the chunks of one document under one chunking configuration."""
    target = paths.chunks_dir / spec_name / f"{doc_id}.json"
    write_json(
        target,
        {
            "spec": spec_name,
            "doc_id": doc_id,
            "chunk_count": len(chunks),
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        },
    )
    return target


def load_chunks(paths: ProjectPaths, spec_name: str, doc_id: str) -> list[Chunk]:
    """Read back the chunks of one document."""
    source = paths.chunks_dir / spec_name / f"{doc_id}.json"
    if not source.is_file():
        raise ArtifactError(f"no chunks for document {doc_id!r} under {spec_name!r}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [Chunk.model_validate(chunk) for chunk in payload["chunks"]]

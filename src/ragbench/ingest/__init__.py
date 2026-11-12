"""Ingestion: reading documents and turning them into pages of text."""

from __future__ import annotations

from .discover import DiscoveredFile, DiscoveryResult, discover_documents, file_digest
from .extract import ExtractionResult, detect_media_type, extract_document
from .normalize import normalize_text
from .runner import IngestSummary, run_ingest

__all__ = [
    "DiscoveredFile",
    "DiscoveryResult",
    "discover_documents",
    "file_digest",
    "ExtractionResult",
    "detect_media_type",
    "extract_document",
    "normalize_text",
    "IngestSummary",
    "run_ingest",
]

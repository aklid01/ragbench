"""Text extraction: PDF, plain text and markdown all become a list of pages.

Page numbers are real for PDFs and absent for text and markdown; the
distinction is carried in the data model rather than faked with a page 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ExtractionError
from ..models import MediaType, Page
from .normalize import normalize_text

__all__ = ["ExtractionResult", "detect_media_type", "extract_document", "SUPPORTED_SUFFIXES"]

SUPPORTED_SUFFIXES: dict[str, MediaType] = {
    ".pdf": "pdf",
    ".txt": "text",
    ".text": "text",
    ".md": "markdown",
    ".markdown": "markdown",
}

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252")


@dataclass
class ExtractionResult:
    """Everything one file yielded."""

    pages: list[Page] = field(default_factory=list)
    title: str = ""
    extractor: str = ""
    warnings: list[str] = field(default_factory=list)


def detect_media_type(path: Path) -> MediaType | None:
    """Return the media type of ``path``, or ``None`` when it is unsupported."""
    return SUPPORTED_SUFFIXES.get(path.suffix.lower())


def extract_document(path: Path) -> ExtractionResult:
    """Extract text from ``path``."""
    media_type = detect_media_type(path)
    if media_type == "pdf":
        return _extract_pdf(path)
    if media_type == "text":
        return _extract_plain(path, extractor="text")
    if media_type == "markdown":
        return _extract_plain(path, extractor="markdown")
    raise ExtractionError(f"unsupported document type: {path.name}")


def _extract_pdf(path: Path) -> ExtractionResult:
    try:
        import pymupdf
    except ImportError as exc:
        raise ExtractionError("pymupdf is required to read PDF files") from exc

    result = ExtractionResult(extractor="pymupdf")
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise ExtractionError(f"could not open PDF {path.name}: {exc}") from exc

    with document:
        if document.needs_pass:
            raise ExtractionError(f"PDF {path.name} is encrypted and cannot be read")
        metadata_title = (document.metadata or {}).get("title") or ""
        result.title = str(metadata_title).strip()
        if document.page_count == 0:
            result.warnings.append("PDF contains no pages")
        for index in range(document.page_count):
            page = document.load_page(index)
            try:
                raw = page.get_text("text")
            except Exception as exc:
                raw = ""
                result.warnings.append(f"page {index + 1} could not be read: {exc}")
            text = normalize_text(raw)
            result.pages.append(Page(page_number=index + 1, text=text, char_count=len(text)))
    if not result.title:
        result.title = path.stem
    return result


def _extract_plain(path: Path, extractor: str) -> ExtractionResult:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"could not read {path.name}: {exc}") from exc

    text = ""
    warnings: list[str] = []
    for encoding in _ENCODINGS:
        try:
            text = raw_bytes.decode(encoding)
            if encoding != "utf-8-sig":
                warnings.append(f"decoded as {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ExtractionError(f"could not decode {path.name} with any known encoding")

    normalized = normalize_text(text)
    result = ExtractionResult(
        pages=[Page(page_number=None, text=normalized, char_count=len(normalized))],
        title=path.stem,
        extractor=extractor,
        warnings=warnings,
    )
    return result

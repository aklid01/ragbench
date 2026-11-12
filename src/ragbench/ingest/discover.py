"""Document discovery and duplicate detection.

Discovery walks a directory and returns the files the workbench can read, in
a stable order. Stability matters: the same directory must always produce the
same document ids, otherwise chunk ids and citations would change between
runs for no reason.

Two kinds of duplicate are detected:

* **byte duplicates** - the same file content under two names. Found here, by
  hashing the file.
* **content duplicates** - different files that hold the same text, for
  example the same report saved as PDF and as extracted text. Found by the
  runner, by hashing the normalised text after extraction.

Neither is reported as an error. A duplicate is a fact about the corpus, and
the ingest summary states which copy was kept and which were skipped.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ..models import MediaType
from .extract import detect_media_type

__all__ = ["DiscoveredFile", "DiscoveryResult", "discover_documents", "file_digest"]


@dataclass(frozen=True)
class DiscoveredFile:
    """A readable file found on disk."""

    path: Path
    media_type: MediaType
    sha256: str
    byte_size: int


@dataclass
class DiscoveryResult:
    """The outcome of walking a directory."""

    files: list[DiscoveredFile] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def discover_documents(directory: Path) -> DiscoveryResult:
    """Find every supported document below ``directory``."""
    result = DiscoveryResult()
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")

    candidates = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and detect_media_type(path) is not None
    )

    seen: dict[str, DiscoveredFile] = {}
    for path in candidates:
        relative = path.relative_to(directory)
        if _is_hidden(relative):
            result.skipped.append(f"{relative} (hidden)")
            continue
        digest = file_digest(path)
        if digest in seen:
            kept = seen[digest]
            result.duplicates.append(
                f"{relative} is a byte for byte copy of {kept.path.relative_to(directory)}"
            )
            continue
        media_type = detect_media_type(path)
        assert media_type is not None
        discovered = DiscoveredFile(
            path=path,
            media_type=media_type,
            sha256=digest,
            byte_size=path.stat().st_size,
        )
        seen[digest] = discovered
        result.files.append(discovered)
    return result

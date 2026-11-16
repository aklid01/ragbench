"""Document discovery and duplicate detection.

Discovery walks a directory in a stable order so the same input always
yields the same document ids. Byte duplicates are found by hashing files;
content duplicates (same text, different format) are found by the runner
after extraction. Neither is an error: the summary states what was kept.
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
        (path, media_type)
        for path in directory.rglob("*")
        if path.is_file() and (media_type := detect_media_type(path)) is not None
    )

    seen: dict[str, DiscoveredFile] = {}
    for path, media_type in candidates:
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
        discovered = DiscoveredFile(
            path=path,
            media_type=media_type,
            sha256=digest,
            byte_size=path.stat().st_size,
        )
        seen[digest] = discovered
        result.files.append(discovered)
    return result

"""Project layout.

The workbench keeps five kinds of state apart from each other, because they
have different lifetimes and different rules about what may be committed:

``data/documents/``      the original files. Never modified, never committed.
``data/extracted/``      extracted text per document, plus the ingest manifest.
``data/chunks/``         chunk files, one directory per chunking configuration.
``data/eval/``           the evaluation dataset. Hand written, versioned, committed.
``artifacts/``           everything reproducible: the vector index, experiment
                         runs, generated reports. Never committed.

The root is the closest ancestor directory that contains a ``pyproject.toml``,
so commands behave the same no matter which directory they run from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ProjectPaths", "find_project_root"]


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` that holds a ``pyproject.toml``."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return current


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved locations of every directory the workbench reads or writes."""

    root: Path
    data_dir_name: str = "data"
    artifacts_dir_name: str = "artifacts"

    @classmethod
    def resolve(cls, root: Path | None = None) -> ProjectPaths:
        return cls(root=find_project_root(root))

    @property
    def data_dir(self) -> Path:
        return self.root / self.data_dir_name

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.artifacts_dir_name

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def experiments_dir(self) -> Path:
        return self.artifacts_dir / "experiments"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"

    def ensure(self) -> ProjectPaths:
        """Create the writable directories. Read-only directories are not created."""
        for directory in (
            self.documents_dir,
            self.extracted_dir,
            self.chunks_dir,
            self.eval_dir,
            self.index_dir,
            self.experiments_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

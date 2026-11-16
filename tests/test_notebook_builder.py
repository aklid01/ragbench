"""The notebook builder script: cells are assembled and written as a notebook."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_notebook.py"


def _load_script(tmp_path: Path) -> Any:
    """Load the builder and point its TARGET into the temporary project."""
    spec = importlib.util.spec_from_file_location("build_notebook_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET = tmp_path / "notebooks" / "evaluation_analysis.ipynb"
    return module


def test_the_builder_writes_a_notebook_with_expected_cells(tmp_path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    module = _load_script(tmp_path)

    module.main()

    target = tmp_path / "notebooks" / "evaluation_analysis.ipynb"
    assert target.is_file()
    notebook = nbformat.read(target, as_version=4)
    kinds = [cell.cell_type for cell in notebook.cells]
    assert kinds[0] == "markdown"
    assert "code" in kinds
    assert len(notebook.cells) == len(module.CELLS)
    assert notebook.metadata["kernelspec"]["name"] == "python3"


def test_the_builder_notebook_cells_load_the_runs_directory(tmp_path: Path) -> None:
    pytest.importorskip("nbformat")
    module = _load_script(tmp_path)

    module.main()

    code_cells = [cell for cell in module.CELLS if cell[0] == "code"]
    joined = "\n".join(source for _, source in code_cells)
    assert '"artifacts" / "experiments" / "runs"' in joined
    assert "read_run_file" in joined
    assert "RetrievalMetrics" in joined

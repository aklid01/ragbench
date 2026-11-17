"""End to end: the experiment grid is planned and compared without any API.

The dry run prints the grid the experiment would execute, which is the
artefact a planning decision can be checked against. Both commands must run
on a project that has not been indexed yet.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragbench.cli import main


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'ragbench'\n", encoding="utf-8")
    return root


def test_the_dry_run_prints_the_full_grid_without_calling_any_api(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)

    assert main(["--root", str(root), "experiment", "--dry-run"]) == 0
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 9
    assert "chunking=large, top_k=3" in output
    assert "chunking=medium, top_k=5" in output
    assert "chunking=small, top_k=8" in output


def test_the_dry_run_json_grid_is_a_complete_configuration_list(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)

    assert main(["--root", str(root), "experiment", "--dry-run", "--json"]) == 0
    grid = json.loads(capsys.readouterr().out)
    assert len(grid) == 9
    specs = {config["spec_name"] for config in grid}
    assert specs == {"large", "medium", "small"}
    top_ks = {config["top_k"] for config in grid}
    assert top_ks == {3, 5, 8}
    for config in grid:
        assert config["name"] == f"{config['spec_name']}-k{config['top_k']}"


def test_the_dry_run_honors_explicit_specs_and_top_ks(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)

    assert (
        main(
            [
                "--root",
                str(root),
                "experiment",
                "--dry-run",
                "--specs",
                "medium-800,large-1200",
                "--top-ks",
                "1,7",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 4
    assert "chunking=medium-800, top_k=1" in output
    assert "chunking=large-1200, top_k=7" in output


def test_compare_on_a_fresh_project_reports_that_nothing_was_saved(tmp_path: Path, capsys) -> None:
    root = _project(tmp_path)

    assert main(["--root", str(root), "compare"]) == 0
    assert "no saved runs to compare" in capsys.readouterr().out

    assert main(["--root", str(root), "compare", "--json"]) == 0
    assert "no saved runs to compare" in capsys.readouterr().out

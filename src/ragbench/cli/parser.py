"""The argument parser: one subcommand per pipeline stage.

Each command defines its own parser next to its handler; this module only
assembles the top level parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import __version__
from .answer import add_ask_parser, add_index_parser, add_search_parser
from .corpus import add_config_parser, add_ingest_parser, add_inspect_parser
from .evaluation import add_compare_parser, add_evaluate_parser, add_experiment_parser

__all__ = ["build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="A modular RAG evaluation workbench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical first run:\n"
            "  ragbench ingest --source data/documents\n"
            "  ragbench inspect documents\n"
            "  ragbench inspect chunks --doc-id <document id>\n"
            "  ragbench index\n"
            '  ragbench ask "what is the net zero target year?"\n'
        ),
    )
    parser.add_argument("--version", action="version", version=f"ragbench {__version__}")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="project root (defaults to the nearest directory holding pyproject.toml)",
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    add_ingest_parser(sub)
    add_inspect_parser(sub)
    add_config_parser(sub)
    add_index_parser(sub)
    add_search_parser(sub)
    add_ask_parser(sub)
    add_evaluate_parser(sub)
    add_experiment_parser(sub)
    add_compare_parser(sub)
    return parser

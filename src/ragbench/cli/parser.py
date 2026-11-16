"""The argument parser: one subcommand per pipeline stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import __version__
from ..chunking.spec import preset_names

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

    ingest = sub.add_parser("ingest", help="extract documents and cut them into chunks")
    ingest.add_argument(
        "--source",
        type=Path,
        default=None,
        help="directory of documents to ingest (default: data/documents)",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="re-extract files that were already ingested, keeping text duplicates",
    )
    ingest.add_argument("--no-chunk", action="store_true", help="extract only, do not cut chunks")
    ingest.add_argument(
        "--preset",
        choices=preset_names(),
        default=None,
        help=f"chunking configuration to use ({', '.join(preset_names())})",
    )
    ingest.add_argument("--chunk-size", type=int, default=None, help="chunk size in tokens")
    ingest.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="chunk overlap in tokens (defaults to a tenth of the chunk size)",
    )
    ingest.add_argument("--json", action="store_true", help="emit the summary as JSON")

    inspect = sub.add_parser("inspect", help="look at what has been ingested")
    inspect_sub = inspect.add_subparsers(dest="inspect_command", required=True, metavar="<what>")
    inspect_sub.add_parser("documents", help="list extracted documents")
    chunks = inspect_sub.add_parser("chunks", help="show the chunks of one document")
    chunks.add_argument("--doc-id", required=True, help="id of the document to show")
    chunks.add_argument(
        "--spec",
        default=None,
        help="chunking configuration the chunks were produced with",
    )

    config = sub.add_parser("config", help="show the effective configuration")
    config.add_argument("--json", action="store_true", help="emit the configuration as JSON")

    index = sub.add_parser("index", help="embed chunks and build the vector index")
    index.add_argument(
        "--spec",
        default=None,
        help="chunking configuration to index (default: the only one available)",
    )
    index.add_argument("--rebuild", action="store_true", help="drop the index and embed again")
    index.add_argument("--json", action="store_true", help="emit the summary as JSON")

    search = sub.add_parser("search", help="retrieve chunks for a query")
    search.add_argument("query", help="the query text")
    search.add_argument("--top-k", type=int, default=None, help="number of chunks to retrieve")
    search.add_argument(
        "--spec",
        default=None,
        help="chunking configuration to search (default: the most recently indexed)",
    )
    search.add_argument("--json", action="store_true", help="emit the results as JSON")

    ask = sub.add_parser("ask", help="answer a question grounded in the indexed corpus")
    ask.add_argument(
        "question",
        nargs="?",
        default=None,
        help="the question (omit it to read questions interactively)",
    )
    ask.add_argument("--top-k", type=int, default=None, help="number of chunks to retrieve")
    ask.add_argument(
        "--spec",
        default=None,
        help="chunking configuration to search (default: the most recently indexed)",
    )
    ask.add_argument("--json", action="store_true", help="emit the answer as JSON")

    evaluate = sub.add_parser("evaluate", help="run the evaluation dataset against the index")
    evaluate.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="path to the evaluation dataset JSON (default: data/eval/dataset.json)",
    )
    evaluate.add_argument("--top-k", type=int, default=None, help="number of chunks to retrieve")
    evaluate.add_argument(
        "--spec",
        default=None,
        help="chunking configuration to search (default: the most recently indexed)",
    )
    evaluate.add_argument("--json", action="store_true", help="emit the summary as JSON")

    experiment = sub.add_parser(
        "experiment", help="run the evaluation dataset over a grid of configurations"
    )
    experiment.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="path to the evaluation dataset JSON (default: data/eval/dataset.json)",
    )
    experiment.add_argument(
        "--specs",
        default=None,
        help="comma separated chunking configurations (default: every indexed preset)",
    )
    experiment.add_argument(
        "--top-ks", default=None, help="comma separated top-k values (default: 3,5,8)"
    )
    experiment.add_argument(
        "--dry-run", action="store_true", help="show the grid without calling any API"
    )
    experiment.add_argument("--json", action="store_true", help="emit progress as JSON")

    compare = sub.add_parser("compare", help="compare saved experiment runs")
    compare.add_argument("--json", action="store_true", help="emit the table as JSON")

    return parser

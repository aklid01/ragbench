"""Command line interface.

One entry point, one command per stage of the pipeline, and nothing hidden.
The commands are safe to run repeatedly: ingestion and chunking are
idempotent, and every run overwrites its own output files.

Errors are reported as ``error: <message>`` on stderr and the process exits
with a non zero status. Credentials are never printed, not even when the
configuration is shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .chunking.pipeline import chunk_documents
from .chunking.spec import ChunkingSpec, get_preset, preset_names
from .chunking.tokenizer import get_counter
from .config import Settings, load_settings
from .errors import RagBenchError
from .ingest.runner import run_ingest
from .store import list_documents, load_chunks, read_manifest

__all__ = ["build_parser", "main"]


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
    ingest.add_argument("--force", action="store_true", help="re-ingest documents already seen")
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

    return parser


def _chunking_spec(args: argparse.Namespace, settings: Settings) -> ChunkingSpec:
    """Pick the chunking configuration for this run.

    The overlap follows the size. When only the size is given on the command
    line the overlap becomes a tenth of it, which is the ratio the built in
    presets use, so a smaller chunk does not end up with an overlap larger
    than the chunk itself.
    """
    if args.preset:
        return get_preset(args.preset)
    size = args.chunk_size or settings.chunking.chunk_size_tokens
    if args.overlap is not None:
        overlap = args.overlap
    elif args.chunk_size is not None:
        overlap = max(1, round(size * 0.1))
    else:
        overlap = settings.chunking.chunk_overlap_tokens
    name = f"custom-{size}-{overlap}"
    return ChunkingSpec(
        name=name,
        chunk_size_tokens=size,
        chunk_overlap_tokens=overlap,
        tokenizer_encoding=settings.chunking.tokenizer_encoding,
    )


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths.ensure()
    source = args.source or paths.documents_dir
    summary = run_ingest(paths, source, force=args.force)

    chunk_summary = None
    if not args.no_chunk:
        spec = _chunking_spec(args, settings)
        counter = get_counter(spec.tokenizer_encoding)
        chunk_summary = chunk_documents(paths, spec, counter)

    if args.json:
        payload = {
            "ingest": {
                "documents": summary.documents,
                "pages": summary.page_count,
                "characters": summary.char_count,
                "duplicates": summary.duplicates,
                "failures": summary.failures,
                "warnings": summary.warnings,
            },
            "chunking": None
            if chunk_summary is None
            else {
                "spec": chunk_summary.spec_name,
                "documents": chunk_summary.documents,
                "chunks": chunk_summary.chunks,
                "tokens": chunk_summary.token_total,
                "largest_chunk_tokens": chunk_summary.largest_chunk_tokens,
                "failures": chunk_summary.failures,
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"ingested: {summary.describe()}")
        for duplicate in summary.duplicates:
            print(f"  duplicate: {duplicate}")
        for failure in summary.failures:
            print(f"  failure: {failure}")
        for warning in summary.warnings:
            print(f"  warning: {warning}")
        if chunk_summary is not None:
            print(f"chunked: {chunk_summary.describe()}")

    return 1 if summary.failures else 0


def _cmd_inspect(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths

    if args.inspect_command == "documents":
        manifest = read_manifest(paths)
        doc_ids = list_documents(paths)
        if not doc_ids:
            print("no documents ingested yet")
            return 0
        print(f"{manifest.get('document_count', len(doc_ids))} documents extracted\n")
        for entry in manifest.get("documents", []):
            pages = entry.get("page_count", "?")
            print(
                f"{entry['doc_id']:<40} {entry['file_name']:<45} pages={pages} "
                f"chars={entry.get('char_count', '?')}"
            )
        return 0

    spec_name = args.spec
    if spec_name is None:
        candidates = sorted(path.name for path in paths.chunks_dir.glob("*") if path.is_dir())
        if not candidates:
            print("no chunks written yet")
            return 0
        spec_name = candidates[0]
        print(f"using chunking configuration {spec_name}")
    try:
        chunks = load_chunks(paths, spec_name, args.doc_id)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for chunk in chunks:
        print(
            f"{chunk.chunk_id}  tokens={chunk.token_count:<5} "
            f"pages={chunk.page_start}-{chunk.page_end}  {chunk.citation_label}"
        )
        print(f"    {chunk.text[:160]}{'...' if len(chunk.text) > 160 else ''}")
    return 0


def _cmd_config(args: argparse.Namespace, settings: Settings) -> int:
    redacted = settings.redacted()
    if args.json:
        print(json.dumps(redacted, indent=2))
    else:
        print(json.dumps(redacted, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(root=args.root)
        if args.command == "ingest":
            return _cmd_ingest(args, settings)
        if args.command == "inspect":
            return _cmd_inspect(args, settings)
        if args.command == "config":
            return _cmd_config(args, settings)
    except RagBenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

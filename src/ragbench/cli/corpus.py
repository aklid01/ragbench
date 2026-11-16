"""The corpus commands: ingest, inspect and config."""

from __future__ import annotations

import argparse
import json
import sys

from ..chunking.pipeline import chunk_documents
from ..chunking.tokenizer import get_counter
from ..config import Settings
from ..ingest.runner import run_ingest
from ..store import list_documents, load_chunks, read_manifest
from .common import _chunking_spec, _resolve_chunk_spec


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
                "unchanged": summary.unchanged,
                "retired": summary.retired,
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
                "pruned": chunk_summary.pruned,
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
        if summary.unchanged:
            print(f"  unchanged: {len(summary.unchanged)} already extracted, skipped")
        for retired in summary.retired:
            print(f"  retired: {retired}")
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
        spec_name = _resolve_chunk_spec(None, settings)
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
    print(json.dumps(settings.redacted(), indent=2))
    return 0

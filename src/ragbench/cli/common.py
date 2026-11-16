"""Argument resolution helpers shared by the command handlers."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from ..chunking.spec import ChunkingSpec, get_preset
from ..config import Settings
from ..errors import RetrievalError
from ..retrieval.indexer import latest_indexed_spec


def _chunking_spec(args: argparse.Namespace, settings: Settings) -> ChunkingSpec:
    """Pick the chunking configuration for this run.

    When only the size is given, the overlap defaults to a tenth of it so it
    never exceeds the chunk.
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


def _resolve_chunk_spec(args_spec: str | None, settings: Settings) -> str:
    """The chunking configuration to operate on.

    An explicit spec wins; otherwise the settings-derived custom spec when it
    exists, then a single available configuration. Several candidates need an
    explicit choice.
    """
    if args_spec:
        return str(args_spec)
    paths = settings.paths
    candidates = sorted(path.name for path in paths.chunks_dir.glob("*") if path.is_dir())
    if not candidates:
        raise RetrievalError("no chunks found; run 'ragbench ingest' first")
    derived = (
        f"custom-{settings.chunking.chunk_size_tokens}-{settings.chunking.chunk_overlap_tokens}"
    )
    if derived in candidates:
        return derived
    if len(candidates) == 1:
        return candidates[0]
    raise RetrievalError(
        f"several chunking configurations exist ({', '.join(candidates)}); choose one with --spec"
    )


def _resolve_spec(args: argparse.Namespace, settings: Settings) -> str:
    if args.spec:
        return str(args.spec)
    spec_name = latest_indexed_spec(settings.paths)
    if spec_name is None:
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    return spec_name


def _dataset_path(args: argparse.Namespace, settings: Settings) -> Path:
    if args.dataset:
        return Path(args.dataset)
    candidate = settings.paths.eval_dir / "dataset.json"
    if not candidate.is_file():
        raise RetrievalError(f"no evaluation dataset at {candidate}; pass one with --dataset")
    return candidate


def _prompt_questions() -> Iterator[str]:
    """Read questions until an empty line or the end of input."""
    while True:
        try:
            line = input("> ")
        except EOFError:
            return
        line = line.strip()
        if not line:
            return
        yield line

"""Command line interface: one command per pipeline stage.

Errors are reported as ``error: <message>`` on stderr with a non zero exit
status. Credentials are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import __version__
from .chunking.pipeline import chunk_documents
from .chunking.spec import ChunkingSpec, get_preset, preset_names
from .chunking.tokenizer import get_counter
from .config import Settings, load_settings
from .embeddings import EmbeddingClient
from .errors import RagBenchError, RetrievalError
from .evaluation import (
    GenerationMetrics,
    OperationalMetrics,
    RetrievalMetrics,
    load_eval_dataset,
)
from .evaluation.runner import run_evaluation, write_run_file
from .generation import AnsweredQuestion, LLMClient, answer_question
from .ingest.runner import run_ingest
from .retrieval.indexer import index_chunks, latest_indexed_spec
from .retrieval.search import search_chunks
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
            "  ragbench index\n"
            "  ragbench ask \"what is the net zero target year?\"\n"
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


def _index_spec(args: argparse.Namespace, settings: Settings) -> str:
    if args.spec:
        return str(args.spec)
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


def _cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    spec_name = _index_spec(args, settings)
    client = EmbeddingClient(settings.embedding)
    try:
        summary = index_chunks(paths, settings.embedding, client, spec_name, rebuild=args.rebuild)
    finally:
        client.close()

    if args.json:
        print(
            json.dumps(
                {
                    "spec": summary.spec_name,
                    "embedded": summary.embedded,
                    "skipped": summary.skipped,
                    "rejected": summary.rejected,
                    "vector_count": summary.vector_count,
                    "dimensions": summary.dimensions,
                    "failures": summary.failures,
                    "warnings": summary.warnings,
                },
                indent=2,
            )
        )
    else:
        print(f"indexed: {summary.describe()}")
        for warning in summary.warnings:
            print(f"  warning: {warning}")
        for failure in summary.failures:
            print(f"  failure: {failure}")
    return 0


def _cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    if not paths.index_dir.is_dir():
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    spec_name = args.spec or latest_indexed_spec(paths)
    if spec_name is None:
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    top_k = args.top_k or settings.retrieval.top_k
    client = EmbeddingClient(settings.embedding)
    try:
        results = search_chunks(paths, settings.embedding, client, spec_name, args.query, top_k)
    finally:
        client.close()

    if args.json:
        payload = {
            "query": args.query,
            "spec": spec_name,
            "top_k": top_k,
            "model": settings.embedding.model,
            "results": [result.model_dump() for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"query: {args.query}  (configuration {spec_name}, top {top_k})")
        for result in results:
            print(
                f"{result.rank}. distance={result.distance:.4f}  {result.citation_label}  "
                f"{result.chunk_id}"
            )
            excerpt = result.text[:160] + ("..." if len(result.text) > 160 else "")
            print(f"    {excerpt}")
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    if not paths.index_dir.is_dir():
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    spec_name = args.spec or latest_indexed_spec(paths)
    if spec_name is None:
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    top_k = args.top_k or settings.retrieval.top_k

    questions = [args.question] if args.question else _prompt_questions()
    embedding = EmbeddingClient(settings.embedding)
    llm = LLMClient(settings.llm)
    try:
        for question in questions:
            result = answer_question(
                paths,
                settings.embedding,
                embedding,
                settings.llm,
                llm,
                spec_name,
                question,
                top_k,
            )
            if args.json:
                print(json.dumps(result.model_dump(mode="json"), indent=2))
            else:
                _print_answer(result, spec_name, top_k)
    finally:
        embedding.close()
        llm.close()
    return 0


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


def _print_answer(result: AnsweredQuestion, spec_name: str, top_k: int) -> None:
    print(f"question: {result.question}  (configuration {spec_name}, top {top_k})")
    if result.abstained:
        print("answer: <abstained> the retrieved passages do not contain enough information")
    else:
        print(f"answer: {result.answer}")
    for citation in result.citations:
        print(f"  [{citation.reference}] {citation.citation_label}  {citation.chunk_id}")
        excerpt = citation.excerpt[:160] + ("..." if len(citation.excerpt) > 160 else "")
        print(f"      {excerpt}")
    if result.invalid_references:
        print(f"  invalid references with no matching passage: {result.invalid_references}")
    if result.total_tokens is not None:
        tokens = (
            f"prompt={result.prompt_tokens} completion={result.completion_tokens} "
            f"total={result.total_tokens}"
        )
    else:
        tokens = "usage not reported"
    print(
        f"model: {result.model or 'unknown'}  tokens: {tokens}  "
        f"retrieval: {result.retrieval_seconds:.2f}s  generation: {result.generation_seconds:.2f}s"
    )


def _dataset_path(args: argparse.Namespace, settings: Settings) -> Path:
    if args.dataset:
        return Path(args.dataset)
    candidate = settings.paths.eval_dir / "dataset.json"
    if not candidate.is_file():
        raise RetrievalError(
            f"no evaluation dataset at {candidate}; pass one with --dataset"
        )
    return candidate


def _resolve_spec(args: argparse.Namespace, settings: Settings) -> str:
    if args.spec:
        return str(args.spec)
    spec_name = latest_indexed_spec(settings.paths)
    if spec_name is None:
        raise RetrievalError("no index has been built yet; run 'ragbench index' first")
    return spec_name


def _cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    paths = settings.paths
    dataset_path = _dataset_path(args, settings)
    dataset = load_eval_dataset(dataset_path)
    spec_name = _resolve_spec(args, settings)
    top_k = args.top_k or settings.retrieval.top_k

    embedding = EmbeddingClient(settings.embedding)
    llm = LLMClient(settings.llm)
    try:
        records, summary = run_evaluation(
            paths,
            settings.embedding,
            embedding,
            settings.llm,
            llm,
            dataset,
            dataset_path,
            spec_name,
            top_k,
        )
    finally:
        embedding.close()
        llm.close()

    run_file = write_run_file(paths, records, summary)
    retrieval = RetrievalMetrics(records)
    generation = GenerationMetrics(records)
    operational = OperationalMetrics(records)

    if args.json:
        print(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "run_file": str(run_file),
                    "config": summary.config.model_dump(),
                    "retrieval": retrieval.as_dict(),
                    "generation": generation.as_dict(),
                    "operational": operational.as_dict(),
                    "errors": [
                        {"case_id": record.case_id, "error": record.error}
                        for record in records
                        if record.error
                    ],
                },
                indent=2,
            )
        )
    else:
        print(f"run {summary.run_id} saved to {run_file}")
        _print_metric_summary(retrieval, generation, operational, summary)
    return 0


def _print_metric_summary(retrieval, generation, operational, summary) -> None:
    print(f"configuration: chunking={summary.config.spec} top_k={summary.config.top_k}")
    print(
        "retrieval: "
        + "  ".join(f"recall@{k}={value:.2f}" for k, value in retrieval.recall.items())
        + "  "
        + "  ".join(f"mrr@{k}={value:.2f}" for k, value in retrieval.mrr.items())
    )
    for name, value in generation.as_dict().items():
        rendered = "n/a" if value is None else f"{value:.2f}"
        print(f"generation: {name}={rendered}")
    print(f"operational: {json.dumps(operational.as_dict())}")


def _cmd_experiment(args: argparse.Namespace, settings: Settings) -> int:
    from .evaluation.experiments import default_grid

    paths = settings.paths
    top_ks = (
        [int(value) for value in args.top_ks.split(",")]
        if args.top_ks
        else [3, 5, 8]
    )
    if args.specs:
        specs = [value.strip() for value in args.specs.split(",") if value.strip()]
    else:
        indexed = sorted(path.name for path in paths.index_dir.glob("*") if path.is_dir())
        specs = indexed or sorted(preset_names())
    grid = default_grid(specs, top_ks)

    if args.dry_run:
        if args.json:
            print(json.dumps([config.model_dump() for config in grid], indent=2))
        else:
            for config in grid:
                print(config.describe())
        return 0

    dataset_path = _dataset_path(args, settings)
    dataset = load_eval_dataset(dataset_path)
    embedding = EmbeddingClient(settings.embedding)
    llm = LLMClient(settings.llm)
    saved: list[str] = []
    try:
        for config in grid:
            if not (paths.index_dir / config.spec_name).is_dir():
                print(f"skipping {config.describe()}: no index for {config.spec_name}")
                continue
            print(f"running {config.describe()} ...")
            records, summary = run_evaluation(
                paths,
                settings.embedding,
                embedding,
                settings.llm,
                llm,
                dataset,
                dataset_path,
                config.spec_name,
                config.top_k,
            )
            run_file = write_run_file(paths, records, summary)
            saved.append(str(run_file))
            retrieval = RetrievalMetrics(records)
            print(
                f"  run {summary.run_id}: recall@5={retrieval.recall[5]:.2f} "
                f"mrr@5={retrieval.mrr[5]:.2f}"
            )
    finally:
        embedding.close()
        llm.close()
    print(f"{len(saved)} runs saved")
    return 0


def _cmd_compare(args: argparse.Namespace, settings: Settings) -> int:
    from .evaluation.runner import list_run_files, read_run_file

    run_files = list_run_files(settings.paths)
    if not run_files:
        print("no saved runs to compare; run 'ragbench evaluate' or 'ragbench experiment' first")
        return 0

    rows = []
    for run_file in run_files:
        summary, records = read_run_file(run_file)
        retrieval = RetrievalMetrics(records)
        generation = GenerationMetrics(records)
        rows.append(
            {
                "run_id": summary.run_id,
                "spec": summary.config.spec,
                "top_k": summary.config.top_k,
                "embedding_model": summary.config.embedding_model,
                "llm_model": summary.config.llm_model,
                "dataset_version": summary.config.dataset_version,
                "retrieval": retrieval.as_dict(),
                "generation": generation.as_dict(),
                "operational": OperationalMetrics(records).as_dict(),
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    header = (
        f"{'run':<28} {'recall@3':>8} {'recall@5':>8} {'recall@8':>8} "
        f"{'mrr@5':>7} {'f1':>6} {'abstain':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        recall = row["retrieval"]["recall"]
        mrr = row["retrieval"]["mrr"]
        f1 = row["generation"]["token_f1_mean"]
        abstain = row["generation"]["abstention_rate_unanswerable"]
        print(
            f"{row['run_id']:<28} {recall.get('3', 0):>8.2f} {recall.get('5', 0):>8.2f} "
            f"{recall.get('8', 0):>8.2f} {mrr.get('5', 0):>7.2f} "
            f"{'n/a' if f1 is None else f'{f1:.2f}':>6} "
            f"{'n/a' if abstain is None else f'{abstain:.2f}':>8}"
        )
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
        if args.command == "index":
            return _cmd_index(args, settings)
        if args.command == "search":
            return _cmd_search(args, settings)
        if args.command == "ask":
            return _cmd_ask(args, settings)
        if args.command == "evaluate":
            return _cmd_evaluate(args, settings)
        if args.command == "experiment":
            return _cmd_experiment(args, settings)
        if args.command == "compare":
            return _cmd_compare(args, settings)
    except RagBenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

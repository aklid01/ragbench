"""The evaluation commands: evaluate, experiment and compare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..chunking import preset_names
from ..config import Settings
from ..embeddings import EmbeddingClient
from ..errors import RetrievalError
from ..evaluation import (
    GenerationMetrics,
    OperationalMetrics,
    RetrievalMetrics,
    RunSummary,
    default_grid,
    list_run_files,
    load_eval_dataset,
    read_run_file,
    run_evaluation,
    write_run_file,
)
from ..generation import LLMClient
from ..retrieval import read_index_manifest
from .common import _dataset_path, _resolve_spec


def add_evaluate_parser(sub: argparse._SubParsersAction) -> None:
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


def add_experiment_parser(sub: argparse._SubParsersAction) -> None:
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


def add_compare_parser(sub: argparse._SubParsersAction) -> None:
    compare = sub.add_parser("compare", help="compare saved experiment runs")
    compare.add_argument("--json", action="store_true", help="emit the table as JSON")


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


def _print_metric_summary(
    retrieval: RetrievalMetrics,
    generation: GenerationMetrics,
    operational: OperationalMetrics,
    summary: RunSummary,
) -> None:
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
    paths = settings.paths
    top_ks = [int(value) for value in args.top_ks.split(",")] if args.top_ks else [3, 5, 8]
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
            try:
                read_index_manifest(paths, config.spec_name)
            except RetrievalError:
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
            if records and summary.errors == len(records):
                print(f"  all {len(records)} cases failed; run not saved: {records[0].error}")
                continue
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
    run_files = list_run_files(settings.paths)
    if not run_files:
        print("no saved runs to compare; run 'ragbench evaluate' or 'ragbench experiment' first")
        return 0

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run_file in run_files:
        summary, records = read_run_file(run_file)
        retrieval = RetrievalMetrics(records)
        generation = GenerationMetrics(records)
        row: dict[str, Any] = {
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
        identity = (
            summary.config.dataset_version,
            summary.config.embedding_model,
            summary.config.llm_model,
        )
        groups.setdefault(identity, []).append(row)

    if args.json:
        print(json.dumps([row for rows in groups.values() for row in rows], indent=2))
        return 0

    for identity, rows in sorted(groups.items()):
        print(f"dataset {identity[0]}  embedding {identity[1]}  llm {identity[2]}")
        header = (
            f"{'run':<28} {'recall@3':>8} {'recall@5':>8} {'recall@8':>8} "
            f"{'mrr@5':>7} {'f1':>6} {'abstain':>8}"
        )
        print(header)
        print("-" * len(header))
        for row in sorted(rows, key=lambda row: (row["spec"], row["top_k"], row["run_id"])):
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
        print()
    return 0

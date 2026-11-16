"""Build notebooks/evaluation_analysis.ipynb programmatically.

The notebook itself contains no results: it loads whatever run files exist
under artifacts/experiments/runs at execution time, so it stays reproducible
as runs are added.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "evaluation_analysis.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# RAGBench evaluation analysis

Loads every saved run under `artifacts/experiments/runs/` and compares the
configurations the experiments were run under. Nothing here is hardcoded:
re-run the notebook after new `ragbench evaluate` / `ragbench experiment`
runs and the tables and plots update.

A run file holds one record per evaluation case: the retrieved chunks, the
answer, the citations and the per-case metrics. See `docs/EXPERIMENTS.md`
for the metric definitions and the conclusions drawn from these numbers.""",
    ),
    (
        "code",
        """from pathlib import Path
import pandas as pd

root = Path.cwd()
while not (root / "pyproject.toml").is_file():
    root = root.parent

RUNS_DIR = root / "artifacts" / "experiments" / "runs"
assert RUNS_DIR.is_dir(), f"no runs directory at {RUNS_DIR}; run 'ragbench evaluate' first"

from ragbench.evaluation.runner import read_run_file
from ragbench.evaluation.metrics import RetrievalMetrics, GenerationMetrics, OperationalMetrics

rows = []
for run_file in sorted(RUNS_DIR.glob("*.json")):
    summary, records = read_run_file(run_file)
    retrieval = RetrievalMetrics(records).as_dict()
    generation = GenerationMetrics(records).as_dict()
    operational = OperationalMetrics(records).as_dict()
    rows.append({
        "run_id": summary.run_id,
        "spec": summary.config.spec,
        "top_k": summary.config.top_k,
        "llm_model": summary.config.llm_model,
        "embedding_model": summary.config.embedding_model,
        "dataset_version": summary.config.dataset_version,
        "started_at": summary.started_at,
        **{f"recall@{k}": v for k, v in retrieval["recall"].items()},
        **{f"mrr@{k}": v for k, v in retrieval["mrr"].items()},
        **generation,
        "errors": operational["errors"],
        "mean_retrieval_s": operational["mean_retrieval_seconds"],
        "mean_generation_s": operational["mean_generation_seconds"],
        "total_tokens": operational["total_tokens"],
    })

runs = pd.DataFrame(rows).sort_values(["spec", "top_k"]).reset_index(drop=True)
print(f"{len(runs)} runs loaded")
runs""",
    ),
    (
        "markdown",
        """## Retrieval performance

Recall@K is the share of answerable cases whose expected passage appears in
the top-K retrieved chunks; MRR@K adds credit for ranking it early. Both use
the manually verified dataset in `data/eval/dataset.json`.""",
    ),
    (
        "code",
        """retrieval_columns = [c for c in runs.columns if c.startswith(("recall@", "mrr@"))]
runs[["run_id", "spec", "top_k", *retrieval_columns]]""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for axis, prefix in zip(axes, ("recall@", "mrr@")):
    columns = [c for c in runs.columns if c.startswith(prefix)]
    for spec, group in runs.groupby("spec"):
        group = group.sort_values("top_k").drop_duplicates("top_k")
        ks = sorted(set(group["top_k"]) | {3, 5, 8})
        means = group.set_index("top_k")[columns].mean(axis=0)
        axis.plot(ks, [means.get(f"{prefix}{k}", None) for k in ks], marker="o", label=spec)
    axis.set_title(f"{prefix.rstrip('@')} by top-K")
    axis.set_xlabel("top-K")
    axis.set_ylabel("score")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.3)
    axis.legend(title="chunking")
fig.tight_layout()
fig.savefig(root / "artifacts" / "reports" / "retrieval_by_topk.png", dpi=150)
plt.show()""",
    ),
    (
        "markdown",
        """## Chunk size comparison

At a fixed top-K, how does the chunking configuration change retrieval and
generation quality? Smaller chunks fragment passages but rank precisely;
larger chunks hold whole passages but dilute the embedding.""",
    ),
    (
        "code",
        """DEFAULT_K = 5
at_k = runs[runs["top_k"] == DEFAULT_K].set_index("spec")
comparison_columns = [c for c in runs.columns if c.startswith(("recall@", "mrr@"))] + [
    "citation_precision", "citation_recall", "faithfulness_mean", "token_f1_mean",
    "abstention_rate_unanswerable",
]
present = [c for c in comparison_columns if c in at_k.columns]
at_k[present]""",
    ),
    (
        "code",
        """metric_columns = [c for c in ("recall@5", "mrr@5", "token_f1_mean") if c in at_k.columns]
if metric_columns:
    axis = at_k[metric_columns].plot(kind="bar", figsize=(8, 4), rot=0)
    axis.set_title(f"chunk size comparison at top_k={DEFAULT_K}")
    axis.set_ylabel("score")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.3, axis="y")
    axis.figure.tight_layout()
    axis.figure.savefig(root / "artifacts" / "reports" / "chunk_size_comparison.png", dpi=150)
    plt.show()""",
    ),
    (
        "markdown",
        """## Top-K comparison

At a fixed chunk size, does retrieving more passages help recall, and what
does it cost in tokens and latency?""",
    ),
    (
        "code",
        """if len(runs["top_k"].unique()) > 1:
    axis = runs.pivot_table(index="top_k", columns="spec", values="recall@5", aggfunc="mean").plot(
        kind="bar", figsize=(8, 4), rot=0
    )
    axis.set_title("recall@5 by top-K and chunk size")
    axis.set_ylabel("recall@5")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.3, axis="y")
    axis.figure.tight_layout()
    axis.figure.savefig(root / "artifacts" / "reports" / "topk_comparison.png", dpi=150)
    plt.show()
else:
    print("only one top-K in the saved runs; run 'ragbench experiment' for the grid")""",
    ),
    (
        "markdown",
        """## Latency and token usage

Operational cost per configuration. Generation latency dominates; retrieval
is one embedding call plus a local vector search.""",
    ),
    (
        "code",
        """if len(runs) > 1 and len(runs["total_tokens"].dropna()) > 1:
    axis = runs.plot.scatter(x="mean_generation_s", y="total_tokens", figsize=(6, 4))
    for _, row in runs.iterrows():
        axis.annotate(f"{row['spec']}-k{row['top_k']}", (row["mean_generation_s"], row["total_tokens"]))
    axis.set_title("cost per configuration")
    axis.set_xlabel("mean generation seconds per case")
    axis.set_ylabel("total tokens for the dataset")
    axis.grid(alpha=0.3)
    axis.figure.tight_layout()
    axis.figure.savefig(root / "artifacts" / "reports" / "cost_per_configuration.png", dpi=150)
    plt.show()
else:
    runs[["run_id", "mean_retrieval_s", "mean_generation_s", "total_tokens"]]""",
    ),
    (
        "markdown",
        """## Per-case errors and abstentions

Every number above is recomputable from the records. The cells below surface
the cases a reviewer should read by hand: failed cases, abstentions on
answerable questions, and answers that were given on unanswerable ones.""",
    ),
    (
        "code",
        """problem_rows = []
for run_file in sorted(RUNS_DIR.glob("*.json")):
    summary, records = read_run_file(run_file)
    for record in records:
        if record.error:
            problem_rows.append({"run_id": summary.run_id, "case_id": record.case_id, "problem": f"error: {record.error}"})
        elif record.answerable and record.abstained:
            problem_rows.append({"run_id": summary.run_id, "case_id": record.case_id, "problem": "abstained on an answerable question"})
        elif not record.answerable and not record.abstained:
            problem_rows.append({"run_id": summary.run_id, "case_id": record.case_id, "problem": "answered an unanswerable question"})

pd.DataFrame(problem_rows, columns=["run_id", "case_id", "problem"]) if problem_rows else print("no problem cases in the saved runs")""",
    ),
    (
        "code",
        """last_records = []
for run_file in sorted(RUNS_DIR.glob("*.json")):
    summary, records = read_run_file(run_file)
    abstained = [record.case_id for record in records if record.abstained]
    if abstained:
        last_records.append({"run_id": summary.run_id, "abstained_cases": ", ".join(abstained)})
pd.DataFrame(last_records, columns=["run_id", "abstained_cases"]) if last_records else print("no abstentions recorded")""",
    ),
]


def main() -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    for kind, source in CELLS:
        cell = new_markdown_cell(source) if kind == "markdown" else new_code_cell(source)
        notebook.cells.append(cell)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()

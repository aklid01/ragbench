"""Evaluation: dataset, runner, metrics and experiment configurations."""

from .dataset import DATASET_VERSION, EvalCase, EvalDataset, load_eval_dataset
from .experiments import DEFAULT_GRID, ExperimentConfig, default_grid
from .metrics import (
    GenerationMetrics,
    OperationalMetrics,
    RetrievalMetrics,
    compute_exact_match,
    compute_faithfulness,
    compute_token_f1,
    contains_expected_text,
    mrr_at_k,
    recall_at_k,
)
from .records import CaseRecord, RetrievedRef
from .runner import RunConfig, RunSummary, run_evaluation

__all__ = [
    "DATASET_VERSION",
    "EvalCase",
    "EvalDataset",
    "load_eval_dataset",
    "ExperimentConfig",
    "DEFAULT_GRID",
    "default_grid",
    "CaseRecord",
    "RetrievedRef",
    "RunConfig",
    "RunSummary",
    "run_evaluation",
    "RetrievalMetrics",
    "GenerationMetrics",
    "OperationalMetrics",
    "recall_at_k",
    "mrr_at_k",
    "compute_token_f1",
    "compute_exact_match",
    "compute_faithfulness",
    "contains_expected_text",
]

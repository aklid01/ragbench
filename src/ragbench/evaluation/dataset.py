"""The evaluation dataset: schema, loading and validation.

A dataset is a JSON file of :class:`EvalCase` records. Each case states the
question, the expected answer, the document that should support it, the chunk
text that should back the expected answer, whether the question is answerable
at all and free-form notes a human verifier wrote. The dataset is the source
of truth for every metric: nothing in the evaluation layer re-derives it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..errors import EvaluationError

__all__ = [
    "DATASET_VERSION",
    "EvalCase",
    "EvalDataset",
    "load_eval_dataset",
]

DATASET_VERSION = "1.0.0"


class EvalCase(BaseModel):
    """One manually verified question with everything a metric needs."""

    id: str = Field(description="stable identifier, unique within the dataset")
    question: str
    expected_answer: str | None = Field(
        default=None,
        description="reference answer, written by a human from the corpus",
    )
    expected_doc: str | None = Field(
        default=None,
        description="file name of the document that supports the expected answer",
    )
    expected_text: str | None = Field(
        default=None,
        description="passage text that must appear in a supporting chunk",
    )
    answerable: bool = Field(default=True, description="false marks an unanswerable question")
    notes: str = Field(default="", description="how this case was verified, and why")

    @model_validator(mode="after")
    def _check(self) -> EvalCase:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if self.answerable and not (self.expected_answer and self.expected_doc):
            raise ValueError(f"answerable case {self.id!r} needs expected_answer and expected_doc")
        if not self.answerable and self.expected_answer is not None:
            raise ValueError(f"unanswerable case {self.id!r} must not carry an expected_answer")
        return self


class EvalDataset(BaseModel):
    """A versioned set of cases, checked for duplicate ids."""

    version: str = DATASET_VERSION
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> EvalDataset:
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        return self

    @property
    def answerable(self) -> list[EvalCase]:
        return [case for case in self.cases if case.answerable]

    @property
    def unanswerable(self) -> list[EvalCase]:
        return [case for case in self.cases if not case.answerable]


def load_eval_dataset(path: Path) -> EvalDataset:
    """Read and validate a dataset file."""
    if not path.is_file():
        raise EvaluationError(f"no evaluation dataset at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EvalDataset.model_validate(payload)
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"evaluation dataset at {path} is not understood: {exc}") from exc

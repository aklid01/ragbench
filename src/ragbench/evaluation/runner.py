"""The evaluation runner: cases in, answers generated, records scored and saved.

For every case the runner runs the same path the ``ask`` command uses
(retrieve, prompt, complete, validate citations, detect abstention), then
scores the outcome against the case's expectations and writes one
:class:`CaseRecord`. Nothing here grades truth: ``token_f1`` and
``exact_match`` are lexical comparisons against the reference answers, and
the saved records keep every answer for human review.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..config import EmbeddingConfig, LLMConfig
from ..embeddings import EmbeddingClient
from ..generation.citations import Citation
from ..generation.client import LLMClient
from ..generation.pipeline import AnsweredQuestion, answer_question
from ..paths import ProjectPaths
from .dataset import DATASET_VERSION, EvalCase, EvalDataset
from .metrics import (
    compute_exact_match,
    compute_faithfulness,
    compute_token_f1,
    contains_expected_text,
)
from .records import CaseRecord, RetrievedRef

__all__ = [
    "RunConfig",
    "RunSummary",
    "run_evaluation",
    "write_run_file",
    "read_run_file",
    "list_run_files",
    "rank_of_expected",
    "record_for_case",
]

RUNS_DIR_NAME = "runs"


class RunConfig(BaseModel):
    """The configuration one evaluation run was made under."""

    spec: str
    top_k: int
    embedding_model: str
    llm_model: str
    dataset_version: str = DATASET_VERSION
    dataset_path: str


class RunSummary(BaseModel):
    """The header of one run file."""

    run_id: str
    started_at: datetime
    finished_at: datetime
    config: RunConfig
    total_cases: int
    errors: int


def run_id_for(prefix: str, started_at: datetime) -> str:
    """A sortable run id: timestamp plus configuration prefix."""
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{prefix}"


def rank_of_expected(retrieved: list[RetrievedRef], expected_text: str | None) -> int | None:
    """The rank of the first retrieved chunk holding the expected passage."""
    if not expected_text:
        return None
    for ref in retrieved:
        if contains_expected_text(ref.text, expected_text):
            return ref.rank
    return None


def _cited_expected(
    citations: list[Citation], retrieved: list[RetrievedRef], expected_text: str | None
) -> bool:
    """True when a validated citation points at a chunk holding the expected passage."""
    if not expected_text:
        return False
    for citation in citations:
        position = citation.reference - 1
        if not (0 <= position < len(retrieved)):
            continue
        if retrieved[position].chunk_id != citation.chunk_id:
            continue
        if contains_expected_text(retrieved[position].text, expected_text):
            return True
    return False


def record_for_case(
    case: EvalCase,
    answered: AnsweredQuestion | None,
    error: str | None = None,
) -> CaseRecord:
    """Build one scored record from one answered question."""
    record = CaseRecord(
        case_id=case.id,
        question=case.question,
        answerable=case.answerable,
        expected_doc=case.expected_doc,
        expected_text=case.expected_text,
        expected_answer=case.expected_answer,
        error=error,
    )
    if answered is None:
        return record

    retrieved = [
        RetrievedRef(
            rank=chunk.rank,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            file_name=chunk.file_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
        )
        for chunk in answered.retrieved
    ]
    record.answer = answered.answer
    record.abstained = answered.abstained
    record.citations = answered.citations
    record.invalid_references = answered.invalid_references
    record.retrieved = retrieved
    record.retrieval_seconds = answered.retrieval_seconds
    record.generation_seconds = answered.generation_seconds
    record.prompt_tokens = answered.prompt_tokens
    record.completion_tokens = answered.completion_tokens
    record.total_tokens = answered.total_tokens

    record.rank_of_expected = rank_of_expected(retrieved, case.expected_text)
    valid_references = {citation.reference for citation in answered.citations}
    if not answered.abstained:
        record.faithfulness = compute_faithfulness(answered.answer, valid_references)

    if case.answerable and case.expected_answer and not answered.abstained:
        record.token_f1 = compute_token_f1(answered.answer, case.expected_answer)
        record.exact_match = compute_exact_match(answered.answer, case.expected_answer)

    if case.expected_text:
        record.cited_expected = _cited_expected(answered.citations, retrieved, case.expected_text)
    return record


def run_evaluation(
    paths: ProjectPaths,
    embedding: EmbeddingConfig,
    embedding_client: EmbeddingClient,
    llm: LLMConfig,
    llm_client: LLMClient,
    dataset: EvalDataset,
    dataset_path: Path,
    spec_name: str,
    top_k: int,
) -> tuple[list[CaseRecord], RunSummary]:
    """Run every case of ``dataset`` and return the scored records and summary."""
    started_at = datetime.now(UTC)
    config = RunConfig(
        spec=spec_name,
        top_k=top_k,
        embedding_model=embedding.model,
        llm_model=llm.model,
        dataset_version=DATASET_VERSION,
        dataset_path=str(dataset_path),
    )
    records: list[CaseRecord] = []
    for case in dataset.cases:
        try:
            answered = answer_question(
                paths,
                embedding,
                embedding_client,
                llm,
                llm_client,
                spec_name,
                case.question,
                top_k,
            )
            record = record_for_case(case, answered)
        except Exception as exc:
            record = record_for_case(case, None, error=str(exc))
        records.append(record)

    finished_at = datetime.now(UTC)
    summary = RunSummary(
        run_id=run_id_for(f"{spec_name}-k{top_k}", started_at),
        started_at=started_at,
        finished_at=finished_at,
        config=config,
        total_cases=len(records),
        errors=sum(1 for record in records if record.error),
    )
    return records, summary


def run_dir(paths: ProjectPaths) -> Path:
    return paths.experiments_dir / RUNS_DIR_NAME


def write_run_file(paths: ProjectPaths, records: list[CaseRecord], summary: RunSummary) -> Path:
    """Persist one run as JSON under artifacts/experiments/runs."""
    target = run_dir(paths) / f"{summary.run_id}.json"
    payload = {
        "summary": summary.model_dump(mode="json"),
        "records": [record.model_dump(mode="json") for record in records],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def read_run_file(path: Path) -> tuple[RunSummary, list[CaseRecord]]:
    """Read one run file back."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = RunSummary.model_validate(payload["summary"])
    records = [CaseRecord.model_validate(record) for record in payload["records"]]
    return summary, records


def list_run_files(paths: ProjectPaths) -> list[Path]:
    """Every saved run file, oldest first."""
    directory = run_dir(paths)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))

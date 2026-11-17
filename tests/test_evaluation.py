"""Dataset schema, loader validation and the runner's record scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_answered_question
from ragbench.errors import EvaluationError
from ragbench.evaluation.dataset import load_eval_dataset
from ragbench.evaluation.experiments import preset_spec


def _write_dataset(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dataset_payload() -> dict:
    return {
        "version": "1.0.0",
        "cases": [
            {
                "id": "q01",
                "question": "What is the net zero year?",
                "expected_answer": "2040",
                "expected_doc": "report.pdf",
                "expected_text": "net zero by 2040",
                "answerable": True,
                "notes": "verified on page 21",
            },
            {
                "id": "q02",
                "question": "What is the CEO's golf handicap?",
                "answerable": False,
                "notes": "not in the corpus",
            },
        ],
    }


def test_dataset_round_trip(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path / "dataset.json", _dataset_payload())
    dataset = load_eval_dataset(path)
    assert dataset.version == "1.0.0"
    assert len(dataset.answerable) == 1
    assert len(dataset.unanswerable) == 1
    assert dataset.cases[0].expected_doc == "report.pdf"


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    payload = _dataset_payload()
    payload["cases"].append(dict(payload["cases"][0]))
    path = _write_dataset(tmp_path / "dataset.json", payload)
    with pytest.raises(EvaluationError, match="duplicate case ids"):
        load_eval_dataset(path)


def test_answerable_case_needs_reference_and_document(tmp_path: Path) -> None:
    payload = _dataset_payload()
    del payload["cases"][0]["expected_doc"]
    path = _write_dataset(tmp_path / "dataset.json", payload)
    with pytest.raises(EvaluationError, match="expected_answer and expected_doc"):
        load_eval_dataset(path)


def test_unanswerable_case_must_not_carry_a_reference(tmp_path: Path) -> None:
    payload = _dataset_payload()
    payload["cases"][1]["expected_answer"] = "3"
    path = _write_dataset(tmp_path / "dataset.json", payload)
    with pytest.raises(EvaluationError, match="must not carry"):
        load_eval_dataset(path)


def test_missing_dataset_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="no evaluation dataset"):
        load_eval_dataset(tmp_path / "absent.json")


def test_record_scoring_end_to_end() -> None:
    from ragbench.evaluation.dataset import EvalCase
    from ragbench.evaluation.runner import record_for_case

    case = EvalCase.model_validate(
        {
            "id": "q01",
            "question": "What is the net zero year?",
            "expected_answer": "The net zero target is 2040",
            "expected_doc": "report.pdf",
            "expected_text": "Progress towards carbon net zero by 2040",
            "answerable": True,
        }
    )
    answered = make_answered_question(
        answer="The company targets net zero by 2040 [1].",
        retrieved_texts=[
            "We have set the following ESG objectives: Progress towards carbon net zero by 2040.",
            "Unrelated text about water consumption and recycling programs.",
        ],
        citation_reference=1,
    )
    record = record_for_case(case, answered)
    assert record.rank_of_expected == 1
    assert record.cited_expected is True
    assert record.faithfulness == 1.0
    assert record.token_f1 is not None and record.token_f1 > 0.5
    assert record.exact_match is False
    assert record.error is None


def test_record_marks_errors_without_answering() -> None:
    from ragbench.evaluation.dataset import EvalCase
    from ragbench.evaluation.runner import record_for_case

    case = EvalCase.model_validate({"id": "q03", "question": "anything", "answerable": False})
    record = record_for_case(case, None, error="endpoint failed")
    assert record.error == "endpoint failed"
    assert record.rank_of_expected is None
    assert record.retrieved == []


def test_rank_of_expected_returns_first_match() -> None:
    from ragbench.evaluation.records import RetrievedRef
    from ragbench.evaluation.runner import rank_of_expected

    retrieved = [
        RetrievedRef(
            rank=1,
            chunk_id="a",
            doc_id="d",
            file_name="f",
            page_start=None,
            page_end=None,
            text="nothing here",
        ),
        RetrievedRef(
            rank=2,
            chunk_id="b",
            doc_id="d",
            file_name="f",
            page_start=None,
            page_end=None,
            text="the net zero commitment covers 2040 and beyond",
        ),
    ]
    assert rank_of_expected(retrieved, "net zero commitment covers 2040") == 2
    assert rank_of_expected(retrieved, "completely different passage") is None
    assert rank_of_expected(retrieved, None) is None


def test_unanswerable_case_is_never_scored_for_f1() -> None:
    from ragbench.evaluation.dataset import EvalCase
    from ragbench.evaluation.runner import record_for_case

    case = EvalCase.model_validate({"id": "q04", "question": "golf?", "answerable": False})
    answered = make_answered_question(
        answer="The handicap is 12 [1].",
        retrieved_texts=["some passage"],
        citation_reference=1,
    )
    record = record_for_case(case, answered)
    assert record.token_f1 is None
    assert record.exact_match is None
    assert record.cited_expected is False


def test_preset_spec_turns_an_unknown_preset_into_a_typed_error() -> None:
    with pytest.raises(EvaluationError) as excinfo:
        preset_spec("no-such-preset")
    assert "unknown chunking preset" in str(excinfo.value)

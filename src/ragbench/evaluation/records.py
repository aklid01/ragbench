"""Records: the serializable outcome of one eval case in one run.

The runner writes one :class:`CaseRecord` per case into a run file. The
record keeps the answer text, every citation and the retrieved chunks, so a
human reviewer can audit any metric number against the raw material, and the
metrics are pure functions over these records that can be recomputed offline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..generation.citations import Citation

__all__ = ["RetrievedRef", "CaseRecord"]


class RetrievedRef(BaseModel):
    """A retrieved chunk as recorded in a run file, trimmed for size."""

    rank: int
    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int | None
    page_end: int | None
    text: str


class CaseRecord(BaseModel):
    """Everything one evaluation case produced, kept for audit and metrics."""

    case_id: str
    question: str
    answerable: bool
    expected_doc: str | None = None
    expected_text: str | None = None
    expected_answer: str | None = None

    answer: str | None = Field(
        default=None, description="the answer text, or the abstention marker"
    )
    abstained: bool = False
    citations: list[Citation] = Field(default_factory=list)
    invalid_references: list[int] = Field(default_factory=list)
    retrieved: list[RetrievedRef] = Field(default_factory=list)

    rank_of_expected: int | None = Field(
        default=None,
        description="1-based rank of the first retrieved chunk containing the expected passage",
    )
    cited_expected: bool = Field(
        default=False,
        description="a validated citation points at a chunk holding the expected passage",
    )
    faithfulness: float | None = Field(
        default=None,
        description="share of answer sentences that carry a validated citation",
    )
    token_f1: float | None = Field(
        default=None, description="token level F1 between the answer and the reference answer"
    )
    exact_match: bool | None = Field(
        default=None, description="answer equals the reference after normalisation"
    )

    error: str | None = None
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

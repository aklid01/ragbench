"""Grounded answering: retrieval, prompt, completion, citations, abstention.

:func:`answer_question` runs one question through the whole stage: retrieve,
build the prompt, complete, then validate citations and abstention. The
result is one :class:`AnsweredQuestion` carrying everything the evaluation
phase will need, so downstream code never has to re-derive provenance.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Sequence

from pydantic import BaseModel, Field

from ..config import EmbeddingConfig, LLMConfig
from ..embeddings import EmbeddingClient
from ..paths import ProjectPaths
from ..retrieval.search import RetrievedChunk, search_chunks
from .citations import Citation, build_citations
from .client import LLMClient, LLMCompletion
from .prompts import build_answer_prompt, detect_abstention

__all__ = ["AnsweredQuestion", "answer_question", "build_answer"]


class AnsweredQuestion(BaseModel):
    """Everything one grounded answer produced, ready for evaluation."""

    question: str
    spec: str = Field(description="chunking configuration the index was built with")
    top_k: int
    answer: str
    abstained: bool
    citations: list[Citation] = Field(default_factory=list)
    invalid_references: list[int] = Field(default_factory=list)
    retrieved: list[RetrievedChunk]
    embedding_model: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    retrieval_seconds: float
    generation_seconds: float
    answered_at: datetime


def build_answer(
    question: str,
    spec_name: str,
    top_k: int,
    retrieved: Sequence[RetrievedChunk],
    completion: LLMCompletion,
    retrieval_seconds: float,
    embedding_model: str,
) -> AnsweredQuestion:
    """Assemble the structured answer from one completion, without any I/O."""
    abstained = detect_abstention(completion.content)
    citations: list[Citation] = []
    invalid_references: list[int] = []
    if not abstained:
        citations, invalid_references = build_citations(completion.content, retrieved)
    return AnsweredQuestion(
        question=question,
        spec=spec_name,
        top_k=top_k,
        answer=completion.content,
        abstained=abstained,
        citations=citations,
        invalid_references=invalid_references,
        retrieved=list(retrieved),
        embedding_model=embedding_model,
        model=completion.model or "",
        finish_reason=completion.finish_reason,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.total_tokens,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=completion.latency_seconds,
        answered_at=datetime.now(UTC),
    )


def answer_question(
    paths: ProjectPaths,
    embedding: EmbeddingConfig,
    embedding_client: EmbeddingClient,
    llm: LLMConfig,
    llm_client: LLMClient,
    spec_name: str,
    question: str,
    top_k: int,
) -> AnsweredQuestion:
    """Retrieve for ``question``, then ground an answer in what came back."""
    started = time.monotonic()
    retrieved = search_chunks(paths, embedding, embedding_client, spec_name, question, top_k)
    retrieval_seconds = time.monotonic() - started

    completion = llm_client.complete(build_answer_prompt(question, retrieved))
    return build_answer(
        question=question,
        spec_name=spec_name,
        top_k=top_k,
        retrieved=retrieved,
        completion=completion,
        retrieval_seconds=retrieval_seconds,
        embedding_model=embedding.model,
    )

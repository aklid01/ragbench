"""Grounded answer generation through a hosted LLM."""

from .citations import EXCERPT_CHAR_LIMIT, Citation, build_citations, extract_references
from .client import LLMClient, LLMCompletion
from .pipeline import AnsweredQuestion, answer_question, build_answer
from .prompts import ABSTENTION_MARKER, build_answer_prompt, detect_abstention

__all__ = [
    "ABSTENTION_MARKER",
    "EXCERPT_CHAR_LIMIT",
    "LLMClient",
    "LLMCompletion",
    "AnsweredQuestion",
    "Citation",
    "answer_question",
    "build_answer",
    "build_answer_prompt",
    "build_citations",
    "detect_abstention",
    "extract_references",
]

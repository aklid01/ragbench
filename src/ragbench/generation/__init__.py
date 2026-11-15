"""Grounded answer generation through a hosted LLM."""

from .client import LLMClient, LLMCompletion
from .prompts import ABSTENTION_MARKER, build_answer_prompt, detect_abstention

__all__ = [
    "ABSTENTION_MARKER",
    "LLMClient",
    "LLMCompletion",
    "build_answer_prompt",
    "detect_abstention",
]

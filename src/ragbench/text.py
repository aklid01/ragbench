"""Small text utilities shared across the generation and evaluation stages."""

from __future__ import annotations

import re

__all__ = ["split_sentences"]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """The non-empty sentences of ``text``, stripped."""
    return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()]

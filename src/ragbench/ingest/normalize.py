"""Text normalisation for extracted text.

Layout noise is removed (line endings, unicode spaces, soft hyphens, control
characters, whitespace runs) and a word broken across a line by a hyphen is
rejoined when the fragment after the hyphen starts with a lower case letter.
Every rule is idempotent.
"""

from __future__ import annotations

import re

__all__ = ["normalize_text"]

_LINE_ENDING = re.compile(r"\r\n|\r")
_UNICODE_SPACE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_RUN = re.compile(r"[ \t]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_SOFT_HYPHEN = "\u00ad"
_BROKEN_WORD = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=[a-z])")


def normalize_text(text: str) -> str:
    """Return ``text`` with layout noise removed and words rejoined."""
    normalized = _LINE_ENDING.sub("\n", text)
    normalized = _UNICODE_SPACE.sub(" ", normalized)
    normalized = normalized.replace(_SOFT_HYPHEN, "")
    normalized = _CONTROL.sub("", normalized)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = _HORIZONTAL_RUN.sub(" ", normalized)
    normalized = _BLANK_RUN.sub("\n\n", normalized)
    normalized = _BROKEN_WORD.sub("", normalized)
    return normalized.strip()

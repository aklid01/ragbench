"""Text normalisation.

Extracted text arrives dirty. A PDF page contains hard line breaks in the
middle of paragraphs, soft hyphens from the typesetter, non breaking spaces
typed by a designer and the occasional control character from the printer.
Chunking that text directly would produce fragments in the wrong places and
token counts that drift between runs.

The rules applied here, in order:

1. line endings become ``\\n``
2. non breaking and other unicode spaces become a plain space
3. soft hyphens are removed
4. control characters other than newline and tab are dropped
5. trailing whitespace is stripped from every line
6. runs of horizontal whitespace collapse to a single space
7. three or more blank lines collapse to two
8. a word broken across a line by a hyphen is rejoined (``sustai-\\ntainable``
   becomes ``sustainable``)

Rule 8 is the only one that changes words rather than layout. It is applied
only when the fragment after the hyphen starts with a lower case letter,
because ``Anti-\\nInflation`` is a real hyphenated compound and joining it
would corrupt the text. Every rule is a pure function of the input, so
normalising twice gives the same result as normalising once.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_text", "normalize_lines"]

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


def normalize_lines(text: str) -> list[str]:
    """Normalise ``text`` and return it as non empty lines."""
    return [line for line in normalize_text(text).split("\n") if line.strip()]


def strip_accents(text: str) -> str:
    """Remove combining marks. Used for keyword matching, never for display."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))

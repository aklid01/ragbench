"""The prompt contract for grounded answers.

Retrieved passages are numbered ``[1]..[K]`` in the order retrieval returned
them; the model must cite those numbers and answer only from the passages.
Abstention is a fixed marker phrase: when the passages do not hold the answer
the model must reply with it alone, and :func:`detect_abstention` looks for
exactly that, so the evaluation metrics get a deterministic signal instead of
a judgment call.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..retrieval.search import RetrievedChunk

__all__ = ["ABSTENTION_MARKER", "build_answer_prompt", "detect_abstention"]

ABSTENTION_MARKER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""\
You answer questions strictly from the retrieved passages you are given.

Rules:
- Use only the numbered passages. Do not use outside knowledge.
- Cite every factual claim with the number of the passage that supports it, \
in brackets like [1] or [2][3].
- Cite only passages that actually support the claim. Never invent references, \
document names or page numbers.
- If the passages do not contain enough information to answer, reply with \
exactly {ABSTENTION_MARKER} and nothing else.
"""


def build_answer_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    """Build the system and user messages for one grounded answer."""
    blocks = []
    for index, chunk in enumerate(retrieved, start=1):
        blocks.append(f"[{index}] {chunk.citation_label}\n{chunk.text}")
    passages = "\n\n".join(blocks) if blocks else "(no passages were retrieved)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n\n{passages}"},
    ]


def detect_abstention(content: str) -> bool:
    """True when the completion is the abstention marker and nothing else.

    Surrounding whitespace, matching quotes and a trailing period are
    tolerated; anything beyond that counts as an answer, not an abstention.
    """
    normalized = content.strip().strip("\"'").rstrip(".").strip()
    return normalized.upper() == ABSTENTION_MARKER

"""Token counting.

``cl100k_base`` is the default encoding and is recorded in every chunk file
so a run can be reproduced later. The counter is a protocol, so a different
tokenizer can be plugged in without touching the chunker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..errors import ChunkingError

__all__ = ["TokenCounter", "TiktokenCounter", "get_counter"]


@runtime_checkable
class TokenCounter(Protocol):
    """The three operations the chunker needs."""

    def count(self, text: str) -> int:
        """Number of tokens in ``text``."""

    def encode(self, text: str) -> list[int]:
        """Token ids for ``text``."""

    def decode(self, tokens: list[int]) -> str:
        """Text for ``tokens``."""


class TiktokenCounter:
    """A :class:`TokenCounter` backed by ``tiktoken``."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ChunkingError(
                "the 'tiktoken' package is required for token based chunking"
            ) from exc
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise ChunkingError(f"unknown tokenizer encoding {encoding_name!r}: {exc}") from exc
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


def get_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """Build the tokenizer named by ``encoding_name``."""
    return TiktokenCounter(encoding_name)

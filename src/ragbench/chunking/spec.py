"""Chunking configurations.

A chunking configuration is a name, a size, an overlap and a tokenizer. The
three presets are the ones the experiment grid compares; a custom
configuration is anything else with the same shape.

Overlap is always a fraction of the size, which keeps the meaning of the
setting stable when the size changes: a reader can compare "small" and
"large" without having to remember two unrelated overlap numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ChunkingError

__all__ = ["ChunkingSpec", "PRESETS", "get_preset", "preset_names"]


@dataclass(frozen=True)
class ChunkingSpec:
    """One chunking configuration."""

    name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    tokenizer_encoding: str = "cl100k_base"

    def validate(self) -> ChunkingSpec:
        if self.chunk_size_tokens <= 0:
            raise ChunkingError("chunk_size_tokens must be positive")
        if self.chunk_overlap_tokens < 0:
            raise ChunkingError("chunk_overlap_tokens must not be negative")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ChunkingError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens "
                f"(got {self.chunk_overlap_tokens} >= {self.chunk_size_tokens})"
            )
        return self


PRESETS: dict[str, ChunkingSpec] = {
    "small": ChunkingSpec("small-500", 500, 50),
    "medium": ChunkingSpec("medium-800", 800, 80),
    "large": ChunkingSpec("large-1200", 1200, 120),
}


def preset_names() -> list[str]:
    """Names of the built in configurations."""
    return sorted(PRESETS)


def get_preset(name: str) -> ChunkingSpec:
    """Return the preset called ``name``."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ChunkingError(
            f"unknown chunking preset {name!r}; known presets: {', '.join(preset_names())}"
        ) from exc

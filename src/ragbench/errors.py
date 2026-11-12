"""Error hierarchy for RAGBench.

Every failure mode the package can raise has a named type. Callers (the CLI,
the pipeline, the evaluator) can therefore distinguish a bad configuration
from a failed API call from a broken document, instead of matching on
exception strings.
"""

from __future__ import annotations

__all__ = [
    "RagBenchError",
    "ConfigError",
    "IngestionError",
    "ExtractionError",
    "ChunkingError",
    "EmbeddingError",
    "RetrievalError",
    "GenerationError",
    "EvaluationError",
]


class RagBenchError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(RagBenchError):
    """Raised when configuration is missing, malformed or contradictory."""


class IngestionError(RagBenchError):
    """Raised when a document cannot be discovered, read or written."""


class ExtractionError(IngestionError):
    """Raised when text cannot be extracted from a document."""


class ChunkingError(RagBenchError):
    """Raised when chunking parameters are invalid or chunking fails."""


class EmbeddingError(RagBenchError):
    """Raised when an embedding request fails or returns an invalid vector."""


class RetrievalError(RagBenchError):
    """Raised when the vector store cannot be queried."""


class GenerationError(RagBenchError):
    """Raised when a generation request fails or returns an invalid payload."""


class EvaluationError(RagBenchError):
    """Raised when an evaluation dataset or metric computation is invalid."""

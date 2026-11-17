"""Central configuration.

Everything the workbench needs to reach an API, read a directory or size a
chunk comes from the ``Settings`` object built here. Precedence, strongest
first: values passed to :func:`load_settings`, real environment variables,
the ``.env`` file next to the project root, the defaults in this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import overload
from urllib.parse import urlparse

from pydantic import BaseModel, Field, SecretStr, model_validator

from .errors import ConfigError
from .paths import ProjectPaths, find_project_root

__all__ = [
    "ENV_PREFIX",
    "DEFAULT_EMBEDDING_BASE_URL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_LLM_BASE_URL",
    "DEFAULT_LLM_MODEL",
    "EmbeddingConfig",
    "LLMConfig",
    "ChunkingConfig",
    "RetrievalConfig",
    "Settings",
    "load_settings",
    "load_env_file",
]

ENV_PREFIX = "RAGBENCH_"

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_EMBEDDING_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-8b"


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file into a mapping.

    Only ``KEY=VALUE`` lines are understood. Blank lines and ``#`` comments
    are skipped, and a value may be wrapped in single or double quotes.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


class _BaseProviderConfig(BaseModel):
    """Settings shared by the embedding and generation clients."""

    base_url: str
    model: str
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    retry_base_delay_seconds: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_base_url(self) -> _BaseProviderConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"base_url must be an absolute http(s) URL, got {self.base_url!r}")
        return self

    def masked_key(self) -> str:
        return "set" if self.api_key else "not set"


class EmbeddingConfig(_BaseProviderConfig):
    """Settings for the embedding client.

    The two input types are optional and are only sent when set, for
    asymmetric models that require being told whether the input is a query
    or a passage. ``max_input_tokens`` is the model's input length cap; when
    set, indexing fails fast instead of trusting the endpoint to truncate
    or refuse overlong chunks.
    """

    query_input_type: str | None = None
    passage_input_type: str | None = None
    expected_dimensions: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=16, gt=0)
    requests_per_minute: float | None = Field(default=30.0, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)


class LLMConfig(_BaseProviderConfig):
    """Settings for the generation client."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1200, gt=0)


class ChunkingConfig(BaseModel):
    """Defaults for the chunker. Sizes are counted in tokens, never characters."""

    chunk_size_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=80, ge=0)
    tokenizer_encoding: str = "cl100k_base"

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens "
                f"(got {self.chunk_overlap_tokens} >= {self.chunk_size_tokens})"
            )
        return self


class RetrievalConfig(BaseModel):
    """Defaults for retrieval."""

    top_k: int = Field(default=5, gt=0)


class Settings(BaseModel):
    """The complete configuration of a RAGBench run."""

    root: Path
    data_dir_name: str = "data"
    artifacts_dir_name: str = "artifacts"
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig

    @property
    def paths(self) -> ProjectPaths:
        return ProjectPaths(
            root=self.root,
            data_dir_name=self.data_dir_name,
            artifacts_dir_name=self.artifacts_dir_name,
        )

    def redacted(self) -> dict[str, object]:
        """A printable view of the configuration with secrets removed."""
        return {
            "root": str(self.root),
            "llm": {
                "base_url": self.llm.base_url,
                "model": self.llm.model,
                "api_key": self.llm.masked_key(),
            },
            "embedding": {
                "base_url": self.embedding.base_url,
                "model": self.embedding.model,
                "api_key": self.embedding.masked_key(),
                "expected_dimensions": self.embedding.expected_dimensions,
                "max_input_tokens": self.embedding.max_input_tokens,
            },
            "chunking": self.chunking.model_dump(),
            "retrieval": self.retrieval.model_dump(),
        }


@overload
def _text(env: Mapping[str, str], name: str) -> str | None: ...


@overload
def _text(env: Mapping[str, str], name: str, default: str) -> str: ...


def _text(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    raw = env.get(f"{ENV_PREFIX}{name}")
    if raw is None:
        return default
    value = raw.strip()
    return value or default


@overload
def _number(env: Mapping[str, str], name: str) -> float | None: ...


@overload
def _number(env: Mapping[str, str], name: str, default: float) -> float: ...


def _number(env: Mapping[str, str], name: str, default: float | None = None) -> float | None:
    raw = _text(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from exc


@overload
def _integer(env: Mapping[str, str], name: str) -> int | None: ...


@overload
def _integer(env: Mapping[str, str], name: str, default: int) -> int: ...


def _integer(env: Mapping[str, str], name: str, default: int | None = None) -> int | None:
    raw = _text(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from exc


def load_settings(
    env: Mapping[str, str] | None = None,
    root: Path | None = None,
    use_env_file: bool = True,
) -> Settings:
    """Build the settings for a run.

    ``env`` overrides the process environment, which is how the tests inject
    configuration without touching the machine.
    """
    resolved_root = root.resolve() if root else find_project_root()
    merged: dict[str, str] = {}
    if use_env_file:
        merged.update(load_env_file(resolved_root / ".env"))
    merged.update(os.environ)
    if env:
        merged.update(env)

    data_dir = _text(merged, "DATA_DIR", "data")
    artifacts_dir = _text(merged, "ARTIFACTS_DIR", "artifacts")
    llm_key = _text(merged, "LLM_API_KEY")
    embedding_key = _text(merged, "EMBEDDING_API_KEY")

    try:
        settings = Settings(
            root=resolved_root,
            data_dir_name=data_dir,
            artifacts_dir_name=artifacts_dir,
            llm=LLMConfig(
                base_url=_text(merged, "LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
                model=_text(merged, "LLM_MODEL", DEFAULT_LLM_MODEL),
                api_key=SecretStr(llm_key) if llm_key else None,
                timeout_seconds=_number(merged, "LLM_TIMEOUT_SECONDS", 60.0),
                max_retries=_integer(merged, "LLM_MAX_RETRIES", 4),
            ),
            embedding=EmbeddingConfig(
                base_url=_text(merged, "EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL),
                model=_text(merged, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                api_key=SecretStr(embedding_key) if embedding_key else None,
                timeout_seconds=_number(merged, "EMBEDDING_TIMEOUT_SECONDS", 60.0),
                max_retries=_integer(merged, "EMBEDDING_MAX_RETRIES", 4),
                query_input_type=_text(merged, "EMBEDDING_QUERY_INPUT_TYPE"),
                passage_input_type=_text(merged, "EMBEDDING_PASSAGE_INPUT_TYPE"),
                expected_dimensions=_integer(merged, "EMBEDDING_DIMENSIONS"),
                batch_size=_integer(merged, "EMBEDDING_BATCH_SIZE", 16),
                requests_per_minute=_number(merged, "REQUESTS_PER_MINUTE", 30.0),
                max_input_tokens=_integer(merged, "EMBEDDING_MAX_INPUT_TOKENS"),
            ),
            chunking=ChunkingConfig(
                chunk_size_tokens=_integer(merged, "CHUNK_SIZE_TOKENS", 800),
                chunk_overlap_tokens=_integer(merged, "CHUNK_OVERLAP_TOKENS", 80),
                tokenizer_encoding=_text(merged, "TOKENIZER_ENCODING", "cl100k_base"),
            ),
            retrieval=RetrievalConfig(top_k=_integer(merged, "TOP_K", 5)),
        )
    except Exception as exc:
        raise ConfigError(str(exc)) from exc

    return settings

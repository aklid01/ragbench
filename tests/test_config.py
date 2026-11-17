"""Configuration loading, validation and secret handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.config import load_env_file, load_settings
from ragbench.errors import ConfigError


def test_defaults_are_used_when_nothing_is_set(settings) -> None:
    assert settings.chunking.chunk_size_tokens == 800
    assert settings.chunking.chunk_overlap_tokens == 80
    assert settings.chunking.tokenizer_encoding == "cl100k_base"
    assert settings.retrieval.top_k == 5
    assert settings.llm.base_url == "http://localhost:11434/v1"
    assert settings.embedding.base_url == "https://integrate.api.nvidia.com/v1"


def test_environment_overrides_defaults(project_root: Path) -> None:
    env = {
        "RAGBENCH_CHUNK_SIZE_TOKENS": "500",
        "RAGBENCH_TOP_K": "8",
        "RAGBENCH_LLM_MODEL": "auto/glm-5.3-flash",
    }
    settings = load_settings(env=env, root=project_root, use_env_file=False)
    assert settings.chunking.chunk_size_tokens == 500
    assert settings.retrieval.top_k == 8
    assert settings.llm.model == "auto/glm-5.3-flash"


def test_env_file_is_read_and_environment_wins(project_root: Path) -> None:
    (project_root / ".env").write_text(
        '# a comment\nRAGBENCH_TOP_K=7\nRAGBENCH_LLM_MODEL="from-file"\n', encoding="utf-8"
    )
    settings = load_settings(env={"RAGBENCH_TOP_K": "9"}, root=project_root)
    assert settings.retrieval.top_k == 9
    assert settings.llm.model == "from-file"


def test_env_file_parser_handles_quotes_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("\n# comment\n\nA=1\nB='two'\nC=\"three\"\nBROKEN\n", encoding="utf-8")
    values = load_env_file(path)
    assert values == {"A": "1", "B": "two", "C": "three"}


def test_overlap_must_be_smaller_than_size(project_root: Path) -> None:
    env = {"RAGBENCH_CHUNK_SIZE_TOKENS": "100", "RAGBENCH_CHUNK_OVERLAP_TOKENS": "100"}
    with pytest.raises(ConfigError):
        load_settings(env=env, root=project_root, use_env_file=False)


def test_negative_size_is_rejected(project_root: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(
            env={"RAGBENCH_CHUNK_SIZE_TOKENS": "0"}, root=project_root, use_env_file=False
        )


def test_non_numeric_value_is_rejected(project_root: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(env={"RAGBENCH_TOP_K": "many"}, root=project_root, use_env_file=False)


def test_base_url_must_be_absolute(project_root: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(
            env={"RAGBENCH_LLM_BASE_URL": "localhost:3425"},
            root=project_root,
            use_env_file=False,
        )


def test_secrets_never_appear_in_prints(project_root: Path) -> None:
    env = {
        "RAGBENCH_LLM_API_KEY": "sk-super-secret-llm",
        "RAGBENCH_EMBEDDING_API_KEY": "nvapi-super-secret-embed",
    }
    settings = load_settings(env=env, root=project_root, use_env_file=False)
    rendered = (
        repr(settings) + repr(settings.redacted()) + str(settings.llm) + str(settings.embedding)
    )
    assert "super-secret" not in rendered
    assert settings.llm.masked_key() == "set"
    assert settings.embedding.masked_key() == "set"


def test_missing_key_is_reported_as_not_set(project_root: Path) -> None:
    settings = load_settings(env={}, root=project_root, use_env_file=False)
    assert settings.llm.masked_key() == "not set"
    assert settings.embedding.masked_key() == "not set"


def test_paths_follow_the_configured_directory_names(project_root: Path) -> None:
    env = {"RAGBENCH_DATA_DIR": "corpus", "RAGBENCH_ARTIFACTS_DIR": "output"}
    settings = load_settings(env=env, root=project_root, use_env_file=False)
    assert settings.paths.documents_dir == project_root / "corpus" / "documents"
    assert settings.paths.index_dir == project_root / "output" / "index"

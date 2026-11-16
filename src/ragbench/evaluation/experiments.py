"""The experiment grid: named configurations over (chunking, top-k).

Each :class:`ExperimentConfig` fixes every variable but the ones under
comparison, so a run of one configuration is comparable with a run of
another. The default grid is the one the plan asks for: three chunk sizes
crossed with three top-k values.
"""

from __future__ import annotations

from itertools import product

from pydantic import BaseModel, Field

from ..chunking.spec import PRESETS, ChunkingSpec, get_preset
from ..errors import EvaluationError

__all__ = ["ExperimentConfig", "DEFAULT_GRID", "default_grid", "resolve_chunking_spec"]


class ExperimentConfig(BaseModel):
    """One point of the experiment grid."""

    name: str
    spec_name: str
    top_k: int = Field(gt=0)

    def describe(self) -> str:
        return f"{self.name} (chunking={self.spec_name}, top_k={self.top_k})"


def resolve_chunking_spec(spec_name: str) -> ChunkingSpec:
    """Resolve a preset name to its spec, raising a clear error otherwise."""
    try:
        return get_preset(spec_name)
    except Exception as exc:
        raise EvaluationError(str(exc)) from exc


def default_grid(
    spec_names: list[str] | None = None, top_ks: list[int] | None = None
) -> list[ExperimentConfig]:
    """The cross product of the chunking presets and the top-k values."""
    specs = spec_names or sorted(PRESETS)
    ks = top_ks or [3, 5, 8]
    return [
        ExperimentConfig(name=f"{spec}-k{k}", spec_name=spec, top_k=k)
        for spec, k in product(specs, ks)
    ]


DEFAULT_GRID = default_grid()

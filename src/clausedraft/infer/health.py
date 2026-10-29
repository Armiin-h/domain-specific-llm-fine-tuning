"""Report whether a drafter is ready to continue a clause."""

from __future__ import annotations

from dataclasses import dataclass

from clausedraft.infer.presets import PRESETS
from clausedraft.infer.types import CLAUSE_TYPES


@dataclass(frozen=True)
class Health:
    ok: bool
    model_loaded: bool
    adapter: str | None
    cuda: bool
    clause_types: int
    presets: tuple[str, ...]


def health_report(*, model_loaded: bool, adapter: str | None, cuda: bool) -> Health:
    return Health(
        ok=model_loaded,
        model_loaded=model_loaded,
        adapter=adapter,
        cuda=cuda,
        clause_types=len(CLAUSE_TYPES),
        presets=tuple(PRESETS),
    )

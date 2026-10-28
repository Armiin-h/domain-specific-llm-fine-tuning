"""Score one continuation against the gold tail."""

from __future__ import annotations

from dataclasses import dataclass

from clausedraft.eval.overlap import word_f1
from clausedraft.eval.register import in_legal_register


@dataclass(frozen=True)
class ContinuationScore:
    text: str
    word_f1: float
    in_register: bool
    nll: float | None = None


@dataclass(frozen=True)
class Comparison:
    clause_type: str
    prefix: str
    gold: str
    base: ContinuationScore
    adapter: ContinuationScore | None


def score_continuation(prediction: str, gold: str, nll: float | None = None) -> ContinuationScore:
    return ContinuationScore(
        text=prediction.strip(),
        word_f1=word_f1(prediction, gold),
        in_register=in_legal_register(prediction),
        nll=nll,
    )


def summarize(rows: list[Comparison]) -> dict:
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    def _rate(flags: list[bool]) -> float | None:
        if not flags:
            return None
        return sum(1 for flag in flags if flag) / len(flags)

    base_nll = [row.base.nll for row in rows if row.base.nll is not None]
    adapter_rows = [row.adapter for row in rows if row.adapter is not None]
    adapter_nll = [row.nll for row in adapter_rows if row.nll is not None]
    return {
        "prompts": len(rows),
        "base_word_f1": _mean([row.base.word_f1 for row in rows]),
        "base_in_register": _rate([row.base.in_register for row in rows]),
        "base_nll": _mean(base_nll),
        "adapter_word_f1": _mean([row.word_f1 for row in adapter_rows]),
        "adapter_in_register": _rate([row.in_register for row in adapter_rows]),
        "adapter_nll": _mean(adapter_nll),
    }

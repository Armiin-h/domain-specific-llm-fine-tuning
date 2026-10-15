"""Load LexGLUE LEDGAR provisions into a stable record."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Provision:
    text: str
    clause_type: str
    split: str


def resolve_clause_type(label: int | str, label_names: Sequence[str]) -> str:
    if isinstance(label, str):
        if not label.strip():
            raise ValueError("clause type label is empty")
        return label
    if isinstance(label, bool) or not isinstance(label, int):
        raise ValueError(f"clause type label must be an int or str, got {type(label).__name__}")
    if label < 0 or label >= len(label_names):
        raise ValueError(f"clause type index {label} is outside 0..{len(label_names) - 1}")
    return label_names[label]


def provisions_from_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    label_names: Sequence[str],
) -> list[Provision]:
    provisions: list[Provision] = []
    for split_name, rows in splits.items():
        if not split_name:
            raise ValueError("split name is empty")
        for row in rows:
            provisions.append(
                Provision(
                    text=str(row["text"]),
                    clause_type=resolve_clause_type(row["label"], label_names),
                    split=split_name,
                )
            )
    return provisions


def load_ledgar(dataset: str, subset: str) -> list[Provision]:
    """Download LexGLUE and return every provision in split order."""
    from datasets import load_dataset

    loaded = load_dataset(dataset, subset)
    label_names = list(loaded["train"].features["label"].names)
    return provisions_from_splits(loaded, label_names)

"""Write train, validation, and held-out test chat files."""

from __future__ import annotations

import json
from pathlib import Path

from clausedraft.data.messages import ChatExample


def write_chat_splits(
    examples: list[ChatExample] | tuple[ChatExample, ...],
    *,
    root: Path,
    train_jsonl: str,
    val_jsonl: str,
    test_jsonl: str,
) -> dict[str, int]:
    paths = {
        "train": root / train_jsonl,
        "validation": root / val_jsonl,
        "test": root / test_jsonl,
    }
    grouped: dict[str, list[ChatExample]] = {split: [] for split in paths}
    for example in examples:
        if example.split not in grouped:
            raise ValueError(f"unknown split: {example.split}")
        grouped[example.split].append(example)

    counts: dict[str, int] = {}
    for split, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in grouped[split]:
                handle.write(json.dumps(example.as_json(), ensure_ascii=False) + "\n")
        counts[split] = len(grouped[split])
    return counts


def format_counts(counts: dict[str, int]) -> str:
    return "\n".join(f"{name}: {counts[name]}" for name in ("train", "validation", "test"))

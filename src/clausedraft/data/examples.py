"""Turn cleaned provisions into chat examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clausedraft.data.corpus import Provision
from clausedraft.data.messages import ChatExample
from clausedraft.data.prefix import split_clause


@dataclass(frozen=True)
class BuildResult:
    examples: tuple[ChatExample, ...]
    skipped: int


def read_provisions(path: Path) -> list[Provision]:
    provisions: list[Provision] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                provisions.append(
                    Provision(
                        text=str(row["text"]),
                        clause_type=str(row["clause_type"]),
                        split=str(row["split"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} is missing {exc.args[0]}") from exc
    return provisions


def example_from_provision(provision: Provision, prefix_fraction: float) -> ChatExample | None:
    parts = split_clause(provision.text, prefix_fraction)
    if parts is None:
        return None
    if f"{parts.prefix} {parts.tail}" != provision.text:
        return None
    return ChatExample(
        split=provision.split,
        clause_type=provision.clause_type,
        prefix=parts.prefix,
        tail=parts.tail,
    )


def build_examples(provisions: list[Provision], prefix_fraction: float) -> BuildResult:
    examples: list[ChatExample] = []
    skipped = 0
    for provision in provisions:
        example = example_from_provision(provision, prefix_fraction)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
    return BuildResult(examples=tuple(examples), skipped=skipped)

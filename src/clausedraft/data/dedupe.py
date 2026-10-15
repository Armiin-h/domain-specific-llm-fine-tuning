"""Drop exact copies and near-copies of the same clause."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from clausedraft.data.corpus import Provision

# Holdout rows win when the same clause was also filed in train or validation.
_SPLIT_RANK = {"test": 0, "validation": 1, "train": 2}

_FINGERPRINT = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DedupeResult:
    kept: tuple[Provision, ...]
    dropped_exact: int
    dropped_cross_split: int
    dropped_near: int

    @property
    def dropped(self) -> int:
        return self.dropped_exact + self.dropped_cross_split + self.dropped_near


def fingerprint(text: str) -> str:
    lowered = _FINGERPRINT.sub(" ", text.lower().replace("\u00a0", " "))
    return _SPACE.sub(" ", lowered).strip()


def drop_duplicates(
    provisions: list[Provision],
    *,
    jaccard_threshold: float = 0.92,
) -> DedupeResult:
    if not 0.5 <= jaccard_threshold < 1:
        raise ValueError("jaccard_threshold must be in [0.5, 1)")
    indexed = list(enumerate(provisions))
    kept, dropped_exact, dropped_cross_split = _drop_exact(indexed)
    kept, dropped_near = _drop_near(kept, jaccard_threshold)
    kept.sort(key=lambda item: item[0])
    return DedupeResult(
        kept=tuple(item[1] for item in kept),
        dropped_exact=dropped_exact,
        dropped_cross_split=dropped_cross_split,
        dropped_near=dropped_near,
    )


def format_dedupe(result: DedupeResult) -> str:
    return (
        f"kept: {len(result.kept)}\n"
        f"dropped_exact: {result.dropped_exact}\n"
        f"dropped_cross_split: {result.dropped_cross_split}\n"
        f"dropped_near: {result.dropped_near}"
    )


def _drop_exact(
    indexed: list[tuple[int, Provision]],
) -> tuple[list[tuple[int, Provision]], int, int]:
    groups: dict[str, list[tuple[int, Provision]]] = defaultdict(list)
    for item in indexed:
        groups[fingerprint(item[1].text)].append(item)

    kept: list[tuple[int, Provision]] = []
    dropped_exact = 0
    dropped_cross_split = 0
    for group in groups.values():
        winner = min(group, key=lambda item: (_rank(item[1]), -len(item[1].text), item[0]))
        kept.append(winner)
        for index, provision in group:
            if index == winner[0]:
                continue
            if provision.split != winner[1].split:
                dropped_cross_split += 1
            else:
                dropped_exact += 1
    return kept, dropped_exact, dropped_cross_split


def _drop_near(
    indexed: list[tuple[int, Provision]],
    threshold: float,
) -> tuple[list[tuple[int, Provision]], int]:
    buckets: dict[tuple[str, str, str], list[tuple[int, Provision]]] = defaultdict(list)
    for item in indexed:
        words = fingerprint(item[1].text).split()
        prefix = " ".join(words[:10])
        buckets[(item[1].split, item[1].clause_type, prefix)].append(item)

    kept: list[tuple[int, Provision]] = []
    dropped = 0
    for bucket in buckets.values():
        ordered = sorted(bucket, key=lambda item: (-len(item[1].text), item[0]))
        kept_sets: list[frozenset[str]] = []
        for item in ordered:
            words = frozenset(fingerprint(item[1].text).split())
            if len(words) >= 15 and any(_jaccard(words, previous) >= threshold for previous in kept_sets):
                dropped += 1
                continue
            kept_sets.append(words)
            kept.append(item)
    return kept, dropped


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection == 0:
        return 0.0
    return intersection / len(left | right)


def _rank(provision: Provision) -> int:
    return _SPLIT_RANK.get(provision.split, 9)

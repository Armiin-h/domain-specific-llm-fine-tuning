"""Keep test-split continuations out of the train and validation files."""

from __future__ import annotations

from dataclasses import dataclass

from clausedraft.data.dedupe import fingerprint
from clausedraft.data.messages import ChatExample


@dataclass(frozen=True)
class LeakageReport:
    dropped_train_full: int
    dropped_train_tail: int
    dropped_validation_full: int
    dropped_validation_tail: int

    @property
    def dropped(self) -> int:
        return (
            self.dropped_train_full
            + self.dropped_train_tail
            + self.dropped_validation_full
            + self.dropped_validation_tail
        )


def remove_holdout_overlap(examples: list[ChatExample] | tuple[ChatExample, ...]) -> tuple[tuple[ChatExample, ...], LeakageReport]:
    """Drop train and validation rows whose full text or gold tail is in test."""
    test_full, test_tails = _test_keys(examples)
    kept: list[ChatExample] = []
    dropped_train_full = 0
    dropped_train_tail = 0
    dropped_validation_full = 0
    dropped_validation_tail = 0
    for example in examples:
        if example.split == "test":
            kept.append(example)
            continue
        if example.split not in {"train", "validation"}:
            raise ValueError(f"unknown split: {example.split}")
        full_key = fingerprint(f"{example.prefix} {example.tail}")
        tail_key = fingerprint(example.tail)
        full_hit = full_key in test_full
        tail_hit = tail_key in test_tails
        if not full_hit and not tail_hit:
            kept.append(example)
            continue
        if example.split == "train" and full_hit:
            dropped_train_full += 1
        elif example.split == "train":
            dropped_train_tail += 1
        elif full_hit:
            dropped_validation_full += 1
        else:
            dropped_validation_tail += 1
    report = LeakageReport(
        dropped_train_full=dropped_train_full,
        dropped_train_tail=dropped_train_tail,
        dropped_validation_full=dropped_validation_full,
        dropped_validation_tail=dropped_validation_tail,
    )
    return tuple(kept), report


def assert_no_holdout_overlap(examples: list[ChatExample] | tuple[ChatExample, ...]) -> None:
    _, report = remove_holdout_overlap(examples)
    if report.dropped:
        raise ValueError(f"holdout overlap remains: {report.dropped} rows")


def format_leakage(report: LeakageReport) -> str:
    return (
        f"dropped_train_full: {report.dropped_train_full}\n"
        f"dropped_train_tail: {report.dropped_train_tail}\n"
        f"dropped_validation_full: {report.dropped_validation_full}\n"
        f"dropped_validation_tail: {report.dropped_validation_tail}"
    )


def _test_keys(examples: list[ChatExample] | tuple[ChatExample, ...]) -> tuple[set[str], set[str]]:
    full: set[str] = set()
    tails: set[str] = set()
    for example in examples:
        if example.split != "test":
            continue
        full.add(fingerprint(f"{example.prefix} {example.tail}"))
        tails.add(fingerprint(example.tail))
    return full, tails

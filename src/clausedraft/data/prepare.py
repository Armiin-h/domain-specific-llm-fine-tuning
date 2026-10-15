"""Run clean, dedupe, and the train-set cap, then write provisions JSONL."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from clausedraft.config import REPO_ROOT
from clausedraft.data.clean import CleanResult, clean_provisions, format_clean
from clausedraft.data.corpus import Provision
from clausedraft.data.dedupe import DedupeResult, drop_duplicates, format_dedupe
from clausedraft.data.sample import stratified_sample
from clausedraft.data.stats import CorpusStats, format_stats, summarize

PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "provisions.jsonl"


@dataclass(frozen=True)
class PrepareResult:
    provisions: tuple[Provision, ...]
    raw_stats: CorpusStats
    clean: CleanResult
    dedupe: DedupeResult
    train_rows: int
    validation_rows: int
    test_rows: int


def prepare_provisions(
    provisions: list[Provision],
    *,
    min_chars: int,
    max_chars: int,
    train_cap: int,
    seed: int,
    jaccard_threshold: float = 0.92,
) -> PrepareResult:
    raw_stats = summarize(provisions, min_chars=min_chars, max_chars=max_chars)
    cleaned = clean_provisions(provisions, min_chars=min_chars, max_chars=max_chars)
    deduped = drop_duplicates(list(cleaned.kept), jaccard_threshold=jaccard_threshold)

    train = [item for item in deduped.kept if item.split == "train"]
    holdout = [item for item in deduped.kept if item.split != "train"]
    sampled_train = stratified_sample(train, train_cap, seed)
    final = tuple(sampled_train + holdout)
    counts = Counter(item.split for item in final)
    return PrepareResult(
        provisions=final,
        raw_stats=raw_stats,
        clean=cleaned,
        dedupe=deduped,
        train_rows=counts.get("train", 0),
        validation_rows=counts.get("validation", 0),
        test_rows=counts.get("test", 0),
    )


def write_provisions(path: Path, provisions: tuple[Provision, ...] | list[Provision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in provisions:
            handle.write(
                json.dumps(
                    {"text": item.text, "clause_type": item.clause_type, "split": item.split},
                    ensure_ascii=False,
                )
                + "\n"
            )


def format_prepare(result: PrepareResult) -> str:
    parts = [
        "raw:",
        format_stats(result.raw_stats),
        "",
        "clean:",
        format_clean(result.clean),
        "",
        "dedupe:",
        format_dedupe(result.dedupe),
        "",
        "final:",
        f"train: {result.train_rows}",
        f"validation: {result.validation_rows}",
        f"test: {result.test_rows}",
    ]
    return "\n".join(parts)

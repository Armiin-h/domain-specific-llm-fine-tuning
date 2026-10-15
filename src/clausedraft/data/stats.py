"""Length and label counts for a list of provisions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from clausedraft.data.corpus import Provision


@dataclass(frozen=True)
class CorpusStats:
    split_counts: dict[str, int]
    label_counts: dict[str, int]
    char_min: int
    char_max: int
    char_mean: float
    char_p50: int
    char_p95: int
    below_min: int
    above_max: int

    @property
    def rows(self) -> int:
        return sum(self.split_counts.values())

    @property
    def label_count(self) -> int:
        return len(self.label_counts)


def summarize(provisions: list[Provision], *, min_chars: int, max_chars: int) -> CorpusStats:
    split_counts = dict(Counter(item.split for item in provisions))
    label_counts = dict(Counter(item.clause_type for item in provisions))
    lengths = [len(item.text) for item in provisions]
    if not lengths:
        return CorpusStats(
            split_counts={},
            label_counts={},
            char_min=0,
            char_max=0,
            char_mean=0.0,
            char_p50=0,
            char_p95=0,
            below_min=0,
            above_max=0,
        )
    return CorpusStats(
        split_counts=split_counts,
        label_counts=label_counts,
        char_min=min(lengths),
        char_max=max(lengths),
        char_mean=sum(lengths) / len(lengths),
        char_p50=_percentile(lengths, 50),
        char_p95=_percentile(lengths, 95),
        below_min=sum(1 for n in lengths if n < min_chars),
        above_max=sum(1 for n in lengths if n > max_chars),
    )


def format_stats(stats: CorpusStats, *, top_n: int = 5) -> str:
    lines = [f"rows: {stats.rows}", "splits:"]
    for name in sorted(stats.split_counts):
        lines.append(f"  {name}: {stats.split_counts[name]}")
    lines.append(f"labels: {stats.label_count}")
    lines.append(
        "chars: "
        f"min={stats.char_min} mean={stats.char_mean:.1f} "
        f"p50={stats.char_p50} p95={stats.char_p95} max={stats.char_max}"
    )
    lines.append(f"outside_length: short={stats.below_min} long={stats.above_max}")
    lines.append("most_common:")
    for name, count in _ranked(stats.label_counts)[:top_n]:
        lines.append(f"  {name}: {count}")
    lines.append("least_common:")
    tail = _ranked(stats.label_counts)[-top_n:]
    for name, count in tail:
        lines.append(f"  {name}: {count}")
    return "\n".join(lines)


def _percentile(lengths: list[int], pct: float) -> int:
    ordered = sorted(lengths)
    index = round((pct / 100) * (len(ordered) - 1))
    return ordered[index]


def _ranked(label_counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))

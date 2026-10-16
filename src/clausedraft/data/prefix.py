"""Cut a provision into a prefix and the gold continuation."""

from __future__ import annotations

from dataclasses import dataclass

MIN_SIDE_CHARS = 24


@dataclass(frozen=True)
class ClauseSplit:
    prefix: str
    tail: str


def split_clause(
    text: str,
    prefix_fraction: float,
    *,
    min_prefix_chars: int = MIN_SIDE_CHARS,
    min_tail_chars: int = MIN_SIDE_CHARS,
) -> ClauseSplit | None:
    """Split on a word boundary near `prefix_fraction` of the character length.

    Returns None when either side would be shorter than its minimum. The join
    `prefix + " " + tail` is the original text.
    """
    if not 0.05 <= prefix_fraction <= 0.8:
        raise ValueError("prefix_fraction must be between 0.05 and 0.8")
    if min_prefix_chars < 1 or min_tail_chars < 1:
        raise ValueError("minimum side lengths must be positive")

    words = text.split(" ")
    if len(words) < 2:
        return None

    target = int(round(len(text) * prefix_fraction))
    best_cut: int | None = None
    best_distance: int | None = None
    prefix_len = 0
    for index, word in enumerate(words[:-1]):
        prefix_len += len(word) if index == 0 else len(word) + 1
        tail_len = len(text) - prefix_len - 1
        if prefix_len < min_prefix_chars or tail_len < min_tail_chars:
            continue
        distance = abs(prefix_len - target)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_cut = index + 1

    if best_cut is None:
        return None
    prefix = " ".join(words[:best_cut])
    tail = " ".join(words[best_cut:])
    return ClauseSplit(prefix, tail)

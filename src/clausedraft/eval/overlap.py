"""Word overlap between a continuation and the gold tail."""

from __future__ import annotations

from collections import Counter


def words(text: str) -> list[str]:
    return [token.strip(".,;:()[]\"'").lower() for token in text.split() if token.strip(".,;:()[]\"'")]


def word_f1(prediction: str, gold: str) -> float:
    predicted = words(prediction)
    reference = words(gold)
    if not predicted or not reference:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

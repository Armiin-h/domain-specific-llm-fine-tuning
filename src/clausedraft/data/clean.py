"""Whitespace, boilerplate, and length filters."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from clausedraft.data.corpus import Provision

_WHITESPACE = re.compile(r"\s+")
_KEY = re.compile(r"[^a-z0-9\s]+")

# Headings and placeholder lines that show up as their own "provision".
_BOILERPLATE_KEYS = frozenset(
    {
        "reserved",
        "intentionally omitted",
        "intentionally left blank",
        "omitted",
        "none",
        "n a",
        "na",
        "tbd",
        "to be determined",
        "see attached",
        "not applicable",
    }
)


@dataclass(frozen=True)
class CleanResult:
    kept: tuple[Provision, ...]
    dropped_blank: int
    dropped_boilerplate: int
    dropped_short: int
    dropped_long: int

    @property
    def dropped(self) -> int:
        return self.dropped_blank + self.dropped_boilerplate + self.dropped_short + self.dropped_long


def normalize_whitespace(text: str) -> str:
    cleaned = text.replace("\u00a0", " ").replace("\u200b", "")
    return _WHITESPACE.sub(" ", cleaned).strip()


def is_boilerplate(text: str) -> bool:
    key = _KEY.sub(" ", text.lower())
    key = _WHITESPACE.sub(" ", key).strip()
    if key in _BOILERPLATE_KEYS:
        return True
    if re.fullmatch(r"(section|article)\s+\d+[a-z]?", key):
        return True
    letters = [char for char in text if char.isalpha()]
    if not letters or len(text) > 120:
        return False
    uppercase = sum(char.isupper() for char in letters)
    return uppercase / len(letters) >= 0.9


def clean_provisions(
    provisions: list[Provision],
    *,
    min_chars: int,
    max_chars: int,
) -> CleanResult:
    kept: list[Provision] = []
    dropped_blank = 0
    dropped_boilerplate = 0
    dropped_short = 0
    dropped_long = 0
    for item in provisions:
        text = normalize_whitespace(item.text)
        if not text:
            dropped_blank += 1
            continue
        if is_boilerplate(text):
            dropped_boilerplate += 1
            continue
        if len(text) < min_chars:
            dropped_short += 1
            continue
        if len(text) > max_chars:
            dropped_long += 1
            continue
        kept.append(replace(item, text=text))
    return CleanResult(
        kept=tuple(kept),
        dropped_blank=dropped_blank,
        dropped_boilerplate=dropped_boilerplate,
        dropped_short=dropped_short,
        dropped_long=dropped_long,
    )


def format_clean(result: CleanResult) -> str:
    return (
        f"kept: {len(result.kept)}\n"
        f"dropped_blank: {result.dropped_blank}\n"
        f"dropped_boilerplate: {result.dropped_boilerplate}\n"
        f"dropped_short: {result.dropped_short}\n"
        f"dropped_long: {result.dropped_long}"
    )

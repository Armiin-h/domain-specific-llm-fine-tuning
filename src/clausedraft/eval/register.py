"""A small check that a continuation stays in contract register."""

from __future__ import annotations

_CHATTY_STARTS = (
    "sure",
    "certainly",
    "of course",
    "here is",
    "here's",
    "as an ai",
    "i cannot",
    "i can't",
    "note that",
)
_LEGAL_WORDS = frozenset(
    {
        "shall",
        "agreement",
        "party",
        "parties",
        "herein",
        "hereof",
        "hereto",
        "pursuant",
        "including",
        "without",
        "thereof",
        "hereby",
    }
)


def in_legal_register(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 24:
        return False
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _CHATTY_STARTS):
        return False
    tokens = [token.strip(".,;:()[]\"'") for token in lowered.split()]
    return any(token in _LEGAL_WORDS for token in tokens)

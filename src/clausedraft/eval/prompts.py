"""Held-out clause prompts for the base-versus-adapter comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FIXED_TYPES = (
    "Governing Laws",
    "Indemnifications",
    "Confidentiality",
    "Notices",
    "Severability",
    "Assignments",
    "Waivers",
    "Terminations",
)

_TYPE_PREFIX = "Clause type: "
_CLAUSE_MARKER = "Complete this clause:\n"


@dataclass(frozen=True)
class PromptExample:
    clause_type: str
    prefix: str
    tail: str


def parse_chat_row(record: dict) -> PromptExample:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("row is missing messages")
    user = messages[1].get("content")
    tail = messages[2].get("content")
    if not isinstance(user, str) or not isinstance(tail, str):
        raise ValueError("user or assistant content is missing")
    if not user.startswith(_TYPE_PREFIX) or _CLAUSE_MARKER not in user:
        raise ValueError("user message is not a clause prompt")
    head, prefix = user.split(_CLAUSE_MARKER, 1)
    clause_type = head.removeprefix(_TYPE_PREFIX).strip()
    if not clause_type or not prefix.strip() or not tail.strip():
        raise ValueError("clause type, prefix, or tail is empty")
    return PromptExample(clause_type=clause_type, prefix=prefix.strip(), tail=tail.strip())


def select_fixed_prompts(path: Path, types: tuple[str, ...] = FIXED_TYPES) -> list[PromptExample]:
    """First test-file row of each requested clause type, in the requested order."""
    if not path.is_file():
        raise FileNotFoundError(path)
    wanted = set(types)
    found: dict[str, PromptExample] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(found) == len(wanted):
                break
            if not line.strip():
                continue
            example = parse_chat_row(json.loads(line))
            if example.clause_type in wanted and example.clause_type not in found:
                found[example.clause_type] = example
    return [found[name] for name in types if name in found]

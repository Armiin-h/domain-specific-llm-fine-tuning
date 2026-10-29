"""A user request to continue one clause."""

from __future__ import annotations

from dataclasses import dataclass

from clausedraft.data.messages import SYSTEM_PROMPT, user_prompt
from clausedraft.infer.types import resolve_clause_type


@dataclass(frozen=True)
class DraftRequest:
    clause_type: str
    prefix: str


def make_request(clause_type: str, prefix: str) -> DraftRequest:
    text = prefix.strip()
    if not text:
        raise ValueError("prefix is empty")
    return DraftRequest(resolve_clause_type(clause_type), text)


def request_messages(request: DraftRequest) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(request.clause_type, request.prefix)},
    ]

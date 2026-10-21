"""Load chat JSONL into a datasets.Dataset."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


def load_chat_split(path: Path, limit: int | None = None) -> Dataset:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number} is missing messages")
            rows.append({"messages": messages})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path} has no rows")
    return Dataset.from_list(rows)

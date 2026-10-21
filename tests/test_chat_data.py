import json

import pytest

from clausedraft.train.chat_data import load_chat_split


def _write(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_chat_split_keeps_messages_and_honors_limit(tmp_path):
    path = tmp_path / "train.jsonl"
    _write(
        path,
        [
            {"messages": [{"role": "user", "content": "one"}, {"role": "assistant", "content": "tail one"}]},
            {"messages": [{"role": "user", "content": "two"}, {"role": "assistant", "content": "tail two"}]},
        ],
    )
    dataset = load_chat_split(path, limit=1)
    assert len(dataset) == 1
    assert dataset[0]["messages"][1]["content"] == "tail one"


def test_load_chat_split_rejects_a_row_without_messages(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "no messages"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="messages"):
        load_chat_split(path)

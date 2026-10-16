import json

import pytest

from clausedraft.data.export import format_counts, write_chat_splits
from clausedraft.data.messages import ChatExample


def test_write_chat_splits_keeps_the_test_split_out_of_train(tmp_path):
    examples = [
        ChatExample("train", "Notices", "The company shall notify", "the other party in writing today."),
        ChatExample("validation", "Waivers", "Each party hereby", "waives trial by jury to the extent allowed."),
        ChatExample("test", "Governing Laws", "This Agreement shall", "be governed by the laws of Delaware today."),
    ]
    counts = write_chat_splits(
        examples,
        root=tmp_path,
        train_jsonl="data/processed/train.jsonl",
        val_jsonl="data/processed/val.jsonl",
        test_jsonl="data/processed/test.jsonl",
    )
    assert counts == {"train": 1, "validation": 1, "test": 1}
    train_line = (tmp_path / "data/processed/train.jsonl").read_text(encoding="utf-8")
    assert "Delaware" not in train_line
    assert "the other party in writing today." in train_line
    val_record = json.loads((tmp_path / "data/processed/val.jsonl").read_text(encoding="utf-8"))
    assert val_record["messages"][2]["content"].startswith("waives trial")
    assert "train: 1" in format_counts(counts)


def test_write_chat_splits_rejects_an_unknown_split(tmp_path):
    examples = [ChatExample("holdout", "Notices", "The company shall notify", "the other party in writing today.")]
    with pytest.raises(ValueError, match="unknown split"):
        write_chat_splits(
            examples,
            root=tmp_path,
            train_jsonl="train.jsonl",
            val_jsonl="val.jsonl",
            test_jsonl="test.jsonl",
        )

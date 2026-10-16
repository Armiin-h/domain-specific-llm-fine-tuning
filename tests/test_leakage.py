import pytest

from clausedraft.data.leakage import assert_no_holdout_overlap, remove_holdout_overlap
from clausedraft.data.messages import ChatExample


def _example(split: str, prefix: str, tail: str, clause_type: str = "Notices") -> ChatExample:
    return ChatExample(split, clause_type, prefix, tail)


def test_remove_holdout_overlap_drops_train_copies_of_test_text():
    shared_prefix = "The company shall notify the other party"
    shared_tail = "in writing at the address listed above today."
    rows = [
        _example("train", shared_prefix, shared_tail),
        _example("train", "Payments are due", "within thirty days of the invoice date here."),
        _example("validation", "Each party hereby waives", shared_tail),
        _example("test", shared_prefix, shared_tail),
    ]
    kept, report = remove_holdout_overlap(rows)
    assert report.dropped_train_full == 1
    assert report.dropped_validation_tail == 1
    assert [item.split for item in kept] == ["train", "test"]
    assert_no_holdout_overlap(kept)


def test_assert_no_holdout_overlap_rejects_a_leaking_tail():
    tail = "be governed by the laws of the State of Delaware today."
    rows = [
        _example("train", "This Agreement shall", tail, "Governing Laws"),
        _example("test", "The contract shall", tail, "Governing Laws"),
    ]
    with pytest.raises(ValueError, match="holdout overlap"):
        assert_no_holdout_overlap(rows)

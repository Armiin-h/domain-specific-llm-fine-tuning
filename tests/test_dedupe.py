import pytest

from clausedraft.data.corpus import Provision
from clausedraft.data.dedupe import drop_duplicates, fingerprint


def _long(suffix: str, stem: str = "term") -> str:
    words = [f"{stem}{index}" for index in range(40)]
    return " ".join(words + [suffix])


def test_fingerprint_ignores_case_and_punctuation():
    assert fingerprint("Hello, World!") == fingerprint("hello world")


def test_exact_copy_in_train_and_test_keeps_the_holdout():
    shared = "The company shall notify the other party in writing at the address above."
    rows = [
        Provision(shared, "Notices", "train"),
        Provision(shared, "Notices", "test"),
        Provision(shared + " Extra sentence for the train copy only.", "Notices", "train"),
    ]
    result = drop_duplicates(rows, jaccard_threshold=0.99)
    assert [item.split for item in result.kept] == ["test", "train"]
    assert result.dropped_cross_split == 1
    assert result.dropped_exact == 0


def test_near_copy_with_the_same_opening_collapses_inside_one_split():
    rows = [
        Provision(_long("delaware"), "Indemnifications", "train"),
        Provision(_long("california"), "Indemnifications", "train"),
        Provision(_long("delaware", stem="notice"), "Notices", "train"),
    ]
    result = drop_duplicates(rows)
    assert result.dropped_near == 1
    assert len(result.kept) == 2
    types = {item.clause_type for item in result.kept}
    assert types == {"Indemnifications", "Notices"}


def test_rejects_threshold_outside_range():
    with pytest.raises(ValueError, match="jaccard_threshold"):
        drop_duplicates([], jaccard_threshold=0.2)

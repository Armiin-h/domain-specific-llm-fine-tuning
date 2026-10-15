import pytest

from clausedraft.data.corpus import (
    Provision,
    provisions_from_splits,
    resolve_clause_type,
)

NAMES = ["Governing Laws", "Notices", "Waivers"]


def test_resolve_clause_type_maps_class_index():
    assert resolve_clause_type(2, NAMES) == "Waivers"
    assert resolve_clause_type("Notices", NAMES) == "Notices"


def test_resolve_clause_type_rejects_bad_index():
    with pytest.raises(ValueError, match="outside"):
        resolve_clause_type(3, NAMES)
    with pytest.raises(ValueError, match="empty"):
        resolve_clause_type("  ", NAMES)


def test_provisions_from_splits_keeps_split_and_label_name():
    splits = {
        "train": [{"text": "The parties agree.", "label": 0}],
        "test": [{"text": "Notices go to the address below.", "label": 1}],
    }
    rows = provisions_from_splits(splits, NAMES)
    assert rows == [
        Provision("The parties agree.", "Governing Laws", "train"),
        Provision("Notices go to the address below.", "Notices", "test"),
    ]

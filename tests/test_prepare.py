import json

from clausedraft.data.corpus import Provision
from clausedraft.data.prepare import format_prepare, prepare_provisions, write_provisions


def _provision(text: str, clause_type: str, split: str) -> Provision:
    return Provision(text, clause_type, split)


def test_prepare_caps_train_and_keeps_the_holdout_copy(tmp_path):
    shared = "The buyer shall pay the seller within thirty days of the invoice date."
    body = "alpha " * 30
    rows = [
        _provision("Reserved.", "General", "train"),
        _provision("too short", "General", "train"),
        _provision(shared, "Payments", "train"),
        _provision(shared, "Payments", "test"),
        _provision(body + "one", "Notices", "train"),
        _provision(body + "two", "Notices", "train"),
        _provision(body + "three", "Waivers", "validation"),
    ]
    result = prepare_provisions(
        rows,
        min_chars=40,
        max_chars=4000,
        train_cap=1,
        seed=1,
        jaccard_threshold=0.99,
    )
    assert result.clean.dropped_boilerplate == 1
    assert result.clean.dropped_short == 1
    assert result.dedupe.dropped_cross_split == 1
    assert result.train_rows == 1
    assert result.test_rows == 1
    assert result.validation_rows == 1
    assert all(item.text != shared or item.split == "test" for item in result.provisions)

    path = tmp_path / "provisions.jsonl"
    write_provisions(path, result.provisions)
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(loaded) == 3
    assert set(loaded[0]) == {"text", "clause_type", "split"}
    assert "train: 1" in format_prepare(result)

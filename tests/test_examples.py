import json

from clausedraft.data.corpus import Provision
from clausedraft.data.examples import build_examples, example_from_provision, read_provisions

CLAUSE = (
    "This Agreement shall be governed by the laws of the State of Delaware "
    "without regard to its conflict of laws principles as they apply here."
)


def test_example_reconstructs_the_provision():
    example = example_from_provision(Provision(CLAUSE, "Governing Laws", "train"), 0.4)
    assert example is not None
    assert f"{example.prefix} {example.tail}" == CLAUSE
    assert example.split == "train"
    record = example.as_json()
    assert record["messages"][1]["content"].endswith(example.prefix)
    assert record["messages"][2]["content"] == example.tail


def test_build_examples_skips_a_clause_that_cannot_be_split():
    rows = [
        Provision(CLAUSE, "Governing Laws", "train"),
        Provision("Too short.", "Notices", "train"),
    ]
    result = build_examples(rows, 0.4)
    assert result.skipped == 1
    assert len(result.examples) == 1


def test_read_provisions_roundtrip(tmp_path):
    path = tmp_path / "provisions.jsonl"
    path.write_text(
        json.dumps({"text": CLAUSE, "clause_type": "Governing Laws", "split": "validation"}) + "\n",
        encoding="utf-8",
    )
    rows = read_provisions(path)
    assert rows == [Provision(CLAUSE, "Governing Laws", "validation")]

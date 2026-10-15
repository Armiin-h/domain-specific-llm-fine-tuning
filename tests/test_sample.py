import pytest

from clausedraft.data.corpus import Provision
from clausedraft.data.sample import stratified_sample


def _rows() -> list[Provision]:
    rows: list[Provision] = []
    for index in range(50):
        rows.append(Provision(f"alpha clause {index} " + "word " * 20, "Alpha", "train"))
    for index in range(5):
        rows.append(Provision(f"beta clause {index} " + "word " * 20, "Beta", "train"))
    rows.append(Provision("gamma clause is the rare one " + "word " * 20, "Gamma", "train"))
    return rows


def test_stratified_sample_keeps_the_rare_type_inside_the_cap():
    sampled = stratified_sample(_rows(), cap=10, seed=42)
    assert len(sampled) == 10
    counts = {name: sum(item.clause_type == name for item in sampled) for name in ("Alpha", "Beta", "Gamma")}
    assert counts["Gamma"] == 1
    assert counts["Beta"] >= 1
    assert counts["Alpha"] >= 1


def test_stratified_sample_is_stable_for_a_seed():
    rows = _rows()
    assert stratified_sample(rows, cap=10, seed=7) == stratified_sample(rows, cap=10, seed=7)
    assert stratified_sample(rows, cap=10, seed=7) != stratified_sample(rows, cap=10, seed=8)


def test_stratified_sample_returns_everything_under_the_cap():
    rows = _rows()[:3]
    assert stratified_sample(rows, cap=10, seed=1) == rows


def test_stratified_sample_rejects_a_zero_cap():
    with pytest.raises(ValueError, match="cap"):
        stratified_sample(_rows(), cap=0, seed=1)

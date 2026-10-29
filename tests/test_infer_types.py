import pytest

from clausedraft.infer.types import CLAUSE_TYPES, resolve_clause_type


def test_catalog_has_the_ledgar_hundred():
    assert len(CLAUSE_TYPES) == 100
    assert len(set(CLAUSE_TYPES)) == 100
    assert "Governing Laws" in CLAUSE_TYPES
    assert "Books" in CLAUSE_TYPES


def test_resolve_clause_type_ignores_case_and_spacing():
    assert resolve_clause_type("  governing   laws ") == "Governing Laws"


def test_resolve_clause_type_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown clause type"):
        resolve_clause_type("Indemnification")

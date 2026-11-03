import pytest

from clausedraft.api.fake import EchoDrafter


def test_echo_drafter_keeps_the_request_and_returns_a_clause():
    result = EchoDrafter().complete("governing laws", "  This Agreement shall be governed  ", preset="draft")
    assert result.clause_type == "Governing Laws"
    assert result.prefix == "This Agreement shall be governed"
    assert result.preset == "draft"
    assert "Delaware" in result.continuation
    health = EchoDrafter().health()
    assert health.ok is True
    assert health.clause_types == 100


def test_echo_drafter_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown clause type"):
        EchoDrafter().complete("Not A Clause", "This Agreement shall be governed")

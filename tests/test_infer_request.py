import pytest

from clausedraft.infer.request import make_request, request_messages


def test_request_uses_the_official_type_and_the_training_prompt():
    request = make_request("notices", "  All notices shall be  ")
    messages = request_messages(request)
    assert request.clause_type == "Notices"
    assert messages[0]["role"] == "system"
    assert "All notices shall be" in messages[1]["content"]
    assert "Clause type: Notices" in messages[1]["content"]


def test_request_rejects_an_empty_prefix():
    with pytest.raises(ValueError, match="prefix is empty"):
        make_request("Notices", "   ")

from fastapi.testclient import TestClient

from clausedraft.api.app import create_app
from clausedraft.api.fake import EchoDrafter


def _client() -> TestClient:
    return TestClient(create_app(EchoDrafter()))


def test_complete_returns_the_continuation():
    response = _client().post(
        "/api/complete",
        json={
            "clause_type": "governing laws",
            "prefix": "  This Agreement shall be governed  ",
            "preset": "greedy",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["clause_type"] == "Governing Laws"
    assert body["prefix"] == "This Agreement shall be governed"
    assert body["preset"] == "greedy"
    assert body["continuation"]


def test_complete_rejects_an_empty_prefix_an_unknown_type_and_a_bad_preset():
    client = _client()
    empty = client.post("/api/complete", json={"clause_type": "Notices", "prefix": "   "})
    assert empty.status_code == 400
    assert empty.json()["detail"] == "prefix is empty"

    unknown = client.post(
        "/api/complete",
        json={"clause_type": "Indemnification", "prefix": "The Company shall indemnify"},
    )
    assert unknown.status_code == 400
    assert "unknown clause type" in unknown.json()["detail"]

    preset = client.post(
        "/api/complete",
        json={"clause_type": "Notices", "prefix": "All notices shall be", "preset": "beam"},
    )
    assert preset.status_code == 400
    assert "unknown preset" in preset.json()["detail"]

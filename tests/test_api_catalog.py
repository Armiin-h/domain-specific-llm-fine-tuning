from fastapi.testclient import TestClient

from clausedraft.api.app import create_app
from clausedraft.api.examples import EXAMPLES
from clausedraft.api.fake import EchoDrafter


def _client() -> TestClient:
    return TestClient(create_app(EchoDrafter()))


def test_health_and_catalog():
    client = _client()
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["clause_types"] == 100
    assert body["presets"] == ["greedy", "draft"]

    types = client.get("/api/types")
    assert types.status_code == 200
    names = types.json()["types"]
    assert len(names) == 100
    assert "Governing Laws" in names

    examples = client.get("/api/examples")
    assert examples.status_code == 200
    listed = examples.json()["examples"]
    assert [item["title"] for item in listed] == [item.title for item in EXAMPLES]

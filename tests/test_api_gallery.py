from fastapi.testclient import TestClient

from clausedraft.api.app import create_app
from clausedraft.api.fake import EchoDrafter
from clausedraft.eval.prompts import FIXED_TYPES


def test_gallery_route_returns_the_saved_comparison():
    response = TestClient(create_app(EchoDrafter())).get("/api/gallery")
    assert response.status_code == 200
    body = response.json()
    assert body["adapter"] == "outputs/smoke"
    assert [row["clause_type"] for row in body["rows"]] == list(FIXED_TYPES)
    assert body["rows"][0]["base"]["text"]
    assert body["rows"][0]["adapter"]["text"]


def test_gallery_route_reports_a_missing_file(tmp_path):
    missing = tmp_path / "gallery.md"
    response = TestClient(create_app(EchoDrafter(), gallery_path=missing)).get("/api/gallery")
    assert response.status_code == 404
    assert response.json()["detail"] == "gallery is not in the tree"


def test_gallery_route_rejects_a_file_that_is_not_a_gallery(tmp_path):
    path = tmp_path / "gallery.md"
    path.write_text("not a gallery\n", encoding="utf-8")
    response = TestClient(create_app(EchoDrafter(), gallery_path=path)).get("/api/gallery")
    assert response.status_code == 400
    assert "model or the decoding line" in response.json()["detail"]

from fastapi.testclient import TestClient

from clausedraft.api.app import create_app
from clausedraft.api.fake import EchoDrafter


def test_index_serves_the_page_and_the_disclaimer():
    client = TestClient(create_app(EchoDrafter()))
    page = client.get("/")
    assert page.status_code == 200
    assert "Drafting demo, not legal advice." in page.text
    assert 'id="prefix"' in page.text
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "Enter the start of a clause." in script.text
    assert "Held-out comparison" in page.text
    assert "/api/gallery" in script.text
    assert "No held-out continuations are saved yet." in script.text
    style = client.get("/static/style.css")
    assert style.status_code == 200

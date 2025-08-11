from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_version():
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "app" in body and "version" in body
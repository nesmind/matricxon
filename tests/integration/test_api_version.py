from fastapi.testclient import TestClient

from app.config import MATRICXON_VERSION


def test_returns_the_version_string(client: TestClient) -> None:
    response = client.get("/api/version")

    assert response.status_code == 200
    assert response.json() == {"version": MATRICXON_VERSION}

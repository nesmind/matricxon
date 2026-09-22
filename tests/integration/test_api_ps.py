from fastapi.testclient import TestClient


def test_ps_returns_empty_when_nothing_loaded(client: TestClient) -> None:
    response = client.get("/api/ps")

    assert response.status_code == 200
    assert response.json() == {"models": []}

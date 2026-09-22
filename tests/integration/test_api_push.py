from fastapi.testclient import TestClient


def test_push_fails_closed_with_a_clear_error(client: TestClient) -> None:
    response = client.post("/api/push", json={"model": "anything:latest"})

    assert response.status_code == 501
    assert "registry" in response.json()["error"]

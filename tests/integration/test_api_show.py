from fastapi.testclient import TestClient

from tests.integration.conftest import FakeInstalledModelFactory


def test_show_known_model(client: TestClient, model_factory: FakeInstalledModelFactory) -> None:
    model_factory.create(tag="ministral-3:3b", capabilities=["completion", "thinking"])

    response = client.post("/api/show", json={"model": "ministral-3:3b"})

    assert response.status_code == 200
    assert response.json()["capabilities"] == ["completion", "thinking", "tools"]
    # The tiny fixture's footprint rounds to 0.0GB at 2 decimals: this proves only that the field is
    # present and computed (not crashing on a real-but-unreadable file); the Ministral-3B check
    # covers a meaningful value.
    assert response.json()["estimated_ram_gb"] >= 0


def test_show_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/api/show", json={"model": "nonexistent:latest"})

    assert response.status_code == 404

from fastapi.testclient import TestClient

from tests.integration.conftest import FakeInstalledModelFactory


def test_show_known_model(client: TestClient, model_factory: FakeInstalledModelFactory) -> None:
    model_factory.create(tag="ministral-3:3b", capabilities=["completion", "thinking"])

    response = client.post("/api/show", json={"model": "ministral-3:3b"})

    assert response.status_code == 200
    assert response.json()["capabilities"] == ["completion", "thinking"]
    # The tiny synthetic fixture's real footprint rounds to 0.0GB at 2 decimals - this only proves
    # the field is present and computed (not still-crashing on a real-but-unreadable file), not a
    # meaningful real-world value; that's what the real Ministral-3B check elsewhere covers.
    assert response.json()["estimated_ram_gb"] >= 0


def test_show_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/api/show", json={"model": "nonexistent:latest"})

    assert response.status_code == 404

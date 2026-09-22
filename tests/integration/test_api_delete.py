from pathlib import Path

from fastapi.testclient import TestClient

from tests.integration.conftest import FakeInstalledModelFactory


def test_delete_removes_model_and_sidecar(
    client: TestClient, model_factory: FakeInstalledModelFactory
) -> None:
    installed = model_factory.create(tag="ministral-3:3b")
    gguf_path = Path(installed.path)

    response = client.request("DELETE", "/api/delete", json={"model": "ministral-3:3b"})

    assert response.status_code == 200
    assert not gguf_path.exists()
    assert client.get("/api/tags").json() == {"models": []}


def test_delete_unknown_model_returns_404(client: TestClient) -> None:
    response = client.request("DELETE", "/api/delete", json={"model": "nonexistent:latest"})

    assert response.status_code == 404

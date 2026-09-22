from fastapi.testclient import TestClient

from tests.integration.conftest import FakeInstalledModelFactory


def test_tags_empty_returns_200(client: TestClient) -> None:
    response = client.get("/api/tags")

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_tags_lists_installed_models(
    client: TestClient, model_factory: FakeInstalledModelFactory
) -> None:
    model_factory.create(tag="hf.co/test-org/test-repo:test-model")

    response = client.get("/api/tags")

    assert response.status_code == 200
    [entry] = response.json()["models"]
    assert entry["name"] == "hf.co/test-org/test-repo:test-model"
    assert entry["capabilities"] == ["completion"]
    assert entry["details"]["family"] == "mistral3"

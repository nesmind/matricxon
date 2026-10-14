"""GET /api/node/load: active replies, loaded models, threads and free memory for routing."""

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def test_idle_node_reports_nothing_active(client: TestClient) -> None:
    body = client.get("/api/node/load").json()

    assert body["active"] == 0 and body["models"] == []
    assert body["threads"] >= 1 and body["node_kind"] == "compute"
    assert body["free_memory_bytes"] is None or body["free_memory_bytes"] > 0


def test_a_loaded_model_is_listed_with_its_size(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    client.post("/api/generate", json={"model": "tiny-mistral3:latest"})  # load only

    models = client.get("/api/node/load").json()["models"]

    assert [(m["name"], m["active"]) for m in models] == [("tiny-mistral3:latest", 0)]
    assert models[0]["size_bytes"] > 0

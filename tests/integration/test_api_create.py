import json

from fastapi.testclient import TestClient


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestCreateHappyPath:
    def test_from_an_existing_tag_streams_a_success_line(
        self, client: TestClient, model_factory
    ) -> None:
        model_factory.create(tag="source:latest")

        response = client.post(
            "/api/create", json={"model": "created:latest", "from": "source:latest"}
        )

        assert response.status_code == 200
        lines = _read_ndjson_lines(response)
        assert lines[-1]["status"] == "success"

        tags = {m["name"] for m in client.get("/api/tags").json()["models"]}
        assert "created:latest" in tags


class TestCreateErrors:
    def test_unknown_from_tag_returns_404_before_any_ndjson_line(self, client: TestClient) -> None:
        response = client.post(
            "/api/create", json={"model": "created:latest", "from": "nonexistent:latest"}
        )

        assert response.status_code == 404
        assert response.json()["error"]

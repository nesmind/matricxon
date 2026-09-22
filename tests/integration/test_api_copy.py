from fastapi.testclient import TestClient


class TestCopyHappyPath:
    def test_duplicates_an_installed_model_under_a_new_tag(
        self, client: TestClient, model_factory, models_dir
    ) -> None:
        model_factory.create(tag="source:latest")

        response = client.post(
            "/api/copy", json={"source": "source:latest", "destination": "dest:latest"}
        )

        assert response.status_code == 200
        tags = {m["name"] for m in client.get("/api/tags").json()["models"]}
        assert {"source:latest", "dest:latest"} <= tags

    def test_copying_over_an_existing_destination_tag_overwrites_it(
        self, client: TestClient, model_factory
    ) -> None:
        model_factory.create(tag="source:latest", capabilities=["completion"])
        model_factory.create(tag="dest:latest", capabilities=["embedding"])

        response = client.post(
            "/api/copy", json={"source": "source:latest", "destination": "dest:latest"}
        )

        assert response.status_code == 200
        show = client.post("/api/show", json={"model": "dest:latest"}).json()
        assert show["capabilities"] == ["completion"]


class TestCopyErrors:
    def test_unknown_source_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/copy", json={"source": "nonexistent:latest", "destination": "dest:latest"}
        )

        assert response.status_code == 404

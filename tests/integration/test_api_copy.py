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
        assert show["capabilities"] == ["completion", "tools"]


class TestCopyErrors:
    def test_unknown_source_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/copy", json={"source": "nonexistent:latest", "destination": "dest:latest"}
        )

        assert response.status_code == 404


class TestCopyIsAnAlias:
    def test_the_original_cannot_be_deleted_while_a_copy_exists_but_the_copy_can(
        self, client: TestClient, model_factory
    ) -> None:
        model_factory.create(tag="source:latest")
        client.post("/api/copy", json={"source": "source:latest", "destination": "dest:latest"})

        refused = client.request("DELETE", "/api/delete", json={"model": "source:latest"})
        assert refused.status_code == 409 and "dest:latest" in refused.json()["error"]

        assert (
            client.request("DELETE", "/api/delete", json={"model": "dest:latest"}).status_code
            == 200
        )
        tags = {m["name"] for m in client.get("/api/tags").json()["models"]}
        assert tags == {"source:latest"}
        assert (
            client.request("DELETE", "/api/delete", json={"model": "source:latest"}).status_code
            == 200
        )

    def test_a_refused_delete_does_not_unload_the_model(
        self, client: TestClient, model_factory
    ) -> None:
        from unittest.mock import patch

        from app.dependencies import get_model_manager

        model_factory.create(tag="source:latest")
        client.post("/api/copy", json={"source": "source:latest", "destination": "dest:latest"})
        with patch.object(get_model_manager(), "unload") as unload:
            refused = client.request("DELETE", "/api/delete", json={"model": "source:latest"})
        assert refused.status_code == 409
        unload.assert_not_called()

"""M10: /api/embed (plural, batched) against a tiny but real bert GGUF - the

batched sibling of test_api_embeddings.py. Each input gets its own
EmbeddingEngine.embed() call under the hood (no real padded batching, see
app/routers/embed_router.py's own docstring), but the wire contract (accept
a string or a list, return a list of vectors) is what's under test here.
"""

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


class TestEmbedHappyPath:
    def test_single_string_input_returns_one_embedding(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        response = client.post("/api/embed", json={"model": "tiny-bert:latest", "input": "hi"})

        assert response.status_code == 200
        embeddings = response.json()["embeddings"]
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 8  # N_EMBD in tiny_gguf_bert.py

    def test_list_input_returns_one_embedding_per_item_in_order(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/embed", json={"model": "tiny-bert:latest", "input": ["hi", "there", "hi"]}
        )

        assert response.status_code == 200
        embeddings = response.json()["embeddings"]
        assert len(embeddings) == 3
        assert embeddings[0] == embeddings[2]  # same text -> same vector

    def test_empty_list_input_returns_no_embeddings(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        response = client.post("/api/embed", json={"model": "tiny-bert:latest", "input": []})

        assert response.status_code == 200
        assert response.json()["embeddings"] == []


class TestEmbedErrors:
    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        response = client.post("/api/embed", json={"model": "nonexistent:latest", "input": "hi"})

        assert response.status_code == 404

    def test_any_input_exceeding_num_ctx_returns_400(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/embed",
            json={
                "model": "tiny-bert:latest",
                "input": ["hi", "a very long prompt with many words in it"],
                "options": {"num_ctx": 1},
            },
        )

        assert response.status_code == 400
        assert "num_ctx" in response.json()["error"]

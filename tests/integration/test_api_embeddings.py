"""M8: /api/embeddings against a tiny but real bert GGUF (see
tests/tiny_gguf_bert.py) - the embeddings equivalent of
test_api_chat_real_pipeline.py. Real GGUFModelLoader/BertArchitecture/
WordPieceTokenizer/EmbeddingEngine, not mocked. See
scripts/manual_embed_check.py for the full real-model proof.
"""

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


class TestEmbeddingsHappyPath:
    def test_returns_a_unit_length_embedding_vector(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/embeddings", json={"model": "tiny-bert:latest", "prompt": "hi"}
        )

        assert response.status_code == 200
        embedding = response.json()["embedding"]
        assert len(embedding) == 8  # N_EMBD in tiny_gguf_bert.py
        norm = sum(x * x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 1e-2  # ModelManager loads in bf16, not exactly 1.0

    def test_same_prompt_is_deterministic(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        body = {"model": "tiny-bert:latest", "prompt": "hi"}
        first = client.post("/api/embeddings", json=body).json()["embedding"]
        second = client.post("/api/embeddings", json=body).json()["embedding"]

        assert first == second


class TestEmbeddingsErrors:
    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/embeddings", json={"model": "nonexistent:latest", "prompt": "hi"}
        )

        assert response.status_code == 404

    def test_prompt_exceeding_num_ctx_returns_400(
        self, client: TestClient, tiny_bert_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/embeddings",
            json={
                "model": "tiny-bert:latest",
                "prompt": "a very long prompt with many words in it",
                "options": {"num_ctx": 1},
            },
        )

        assert response.status_code == 400
        assert "num_ctx" in response.json()["error"]

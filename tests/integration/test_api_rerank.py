"""POST /api/rerank against a tiny qwen3 reranker GGUF (rank pooling, `cls.output.weight`), and the
guard that keeps chat away from it: before, such a file loaded as a chat model and replied with
meaningless text."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from tests.tiny_gguf_qwen3 import build_tiny_qwen3_gguf

TAG = "tiny-rerank:latest"
DOCS = ["the cat sat", "dogs bark loudly", "the cat sat"]


@pytest.fixture
def tiny_reranker_model(models_dir: Path) -> InstalledModel:
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-rerank"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = build_tiny_qwen3_gguf(repo_dir / "tiny.gguf", score_labels=["yes", "no"])
    # As a pull records it: name-based inference says "completion".
    installed = InstalledModel(
        tag=TAG,
        path=str(gguf_path),
        architecture="qwen3",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="qwen3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))
    return installed


def _rerank(client: TestClient, **extra: object) -> dict:
    body = {"model": TAG, "query": "cat", "documents": DOCS, **extra}
    response = client.post("/api/rerank", json=body)
    assert response.status_code == 200, response.text
    return response.json()


class TestRerank:
    def test_scores_every_document_best_first(
        self, client: TestClient, tiny_reranker_model: InstalledModel
    ) -> None:
        results = _rerank(client)["results"]

        assert sorted(r["index"] for r in results) == [0, 1, 2]
        scores = [r["relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_identical_documents_get_identical_scores(
        self, client: TestClient, tiny_reranker_model: InstalledModel
    ) -> None:
        by_index = {r["index"]: r["relevance_score"] for r in _rerank(client)["results"]}
        assert by_index[0] == pytest.approx(by_index[2])

    def test_top_n_and_return_documents(
        self, client: TestClient, tiny_reranker_model: InstalledModel
    ) -> None:
        results = _rerank(client, top_n=1, return_documents=True)["results"]

        assert len(results) == 1
        assert results[0]["document"] == DOCS[results[0]["index"]]

    def test_tags_report_only_the_rerank_capability(
        self, client: TestClient, tiny_reranker_model: InstalledModel
    ) -> None:
        models = client.get("/api/tags").json()["models"]
        entry = next(m for m in models if m["name"] == TAG)
        assert entry["capabilities"] == ["rerank"]


class TestWrongModelKind:
    def test_chat_with_a_reranker_is_refused(
        self, client: TestClient, tiny_reranker_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={"model": TAG, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert response.status_code == 400
        assert "reranking model" in response.json()["error"]

    def test_rerank_with_a_chat_model_is_refused(
        self, client: TestClient, tiny_qwen3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/rerank",
            json={"model": "tiny-qwen3:latest", "query": "q", "documents": ["d"]},
        )
        assert response.status_code == 400
        assert "not a reranking model" in response.json()["error"]

    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/rerank", json={"model": "nope:latest", "query": "q", "documents": ["d"]}
        )
        assert response.status_code == 404

    def test_an_overlong_prompt_returns_400(
        self, client: TestClient, tiny_reranker_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/rerank",
            json={"model": TAG, "query": "q", "documents": ["d" * 200], "max_tokens": 64},
        )
        assert response.status_code == 400

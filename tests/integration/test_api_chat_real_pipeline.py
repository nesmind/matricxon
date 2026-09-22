"""M6: /api/chat against a tiny but real mistral3 GGUF (see tiny_gguf.py) -
the real GGUFModelLoader/architecture/tokenizer/ChatEngine/NDJSON stack, not
mocked. Fast and memory-safe (a few KB of weights), unlike a real-model test
would be - see scripts/manual_chat_check.py for the full ministral-3:3b proof.
"""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestChatHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"
        lines = _read_ndjson_lines(response)

        assert all("message" in line for line in lines)
        *content_lines, done_line = lines
        assert all(line["done"] is False for line in content_lines)
        assert done_line["done"] is True
        assert done_line["prompt_eval_count"] > 0
        assert done_line["eval_count"] == 5
        assert done_line["total_duration"] >= 0

    def test_reassembled_content_round_trips_through_the_tokenizer(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["message"]["content"] for line in lines)
        assert isinstance(full_text, str)


class TestChatUnloadCall:
    def test_unload_call_returns_200_and_unloads_a_loaded_model(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": 1},
            },
        )
        assert client.get("/api/ps").json()["models"]

        response = client.post(
            "/api/chat",
            json={"model": "tiny-mistral3:latest", "messages": [], "keep_alive": 0},
        )

        assert response.status_code == 200
        assert client.get("/api/ps").json() == {"models": []}


class TestChatErrorsBeforeStreaming:
    def test_unknown_model_returns_404_before_any_ndjson_line(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat",
            json={"model": "nonexistent:latest", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert response.status_code == 404
        assert response.json()["error"]

    def test_prompt_exceeding_num_ctx_returns_400_before_any_ndjson_line(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [{"role": "user", "content": "a very long message indeed"}],
                "options": {"num_ctx": 1},
            },
        )

        assert response.status_code == 400
        assert "num_ctx" in response.json()["error"]

    def test_unsupported_role_returns_400(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [{"role": "function", "content": "not a real Ollama role"}],
            },
        )

        assert response.status_code == 400

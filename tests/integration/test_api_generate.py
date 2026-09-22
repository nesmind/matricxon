"""M10: /api/generate against a tiny but real mistral3 GGUF (see

tests/tiny_gguf.py) - same real GGUFModelLoader/architecture/tokenizer/
ChatEngine/NDJSON stack as test_api_chat_real_pipeline.py, just reached via
the raw-prompt endpoint instead of the chat-message one.
"""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestGenerateHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/generate",
            json={
                "model": "tiny-mistral3:latest",
                "prompt": "hi",
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"
        lines = _read_ndjson_lines(response)

        assert all("response" in line for line in lines)
        *content_lines, done_line = lines
        assert all(line["done"] is False for line in content_lines)
        assert done_line["done"] is True
        assert done_line["prompt_eval_count"] > 0
        assert done_line["eval_count"] == 5
        assert done_line["total_duration"] >= 0

    def test_reassembled_content_is_a_string(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/generate",
            json={
                "model": "tiny-mistral3:latest",
                "prompt": "hi",
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["response"] for line in lines)
        assert isinstance(full_text, str)


class TestGenerateUnloadCall:
    def test_empty_prompt_with_keep_alive_zero_unloads_without_hanging(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        client.post(
            "/api/generate",
            json={
                "model": "tiny-mistral3:latest",
                "prompt": "hi",
                "options": {"num_predict": 1},
            },
        )
        assert client.get("/api/ps").json()["models"]

        response = client.post(
            "/api/generate",
            json={"model": "tiny-mistral3:latest", "prompt": "", "keep_alive": 0},
        )

        assert response.status_code == 200
        assert response.json() == {}
        assert client.get("/api/ps").json() == {"models": []}


class TestGenerateErrorsBeforeStreaming:
    def test_unknown_model_returns_404_before_any_ndjson_line(self, client: TestClient) -> None:
        response = client.post(
            "/api/generate", json={"model": "nonexistent:latest", "prompt": "hi"}
        )

        assert response.status_code == 404
        assert response.json()["error"]

    def test_prompt_exceeding_num_ctx_returns_400_before_any_ndjson_line(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/generate",
            json={
                "model": "tiny-mistral3:latest",
                "prompt": "a very long prompt with many words in it",
                "options": {"num_ctx": 1},
            },
        )

        assert response.status_code == 400
        assert "num_ctx" in response.json()["error"]

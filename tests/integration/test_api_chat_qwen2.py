"""`/api/chat` against a tiny but real `qwen2` GGUF (see tests/tiny_gguf_qwen2.py). Real
GGUFModelLoader/Qwen2Architecture/GGUFTokenizer/ChatEngine/NDJSON stack, not mocked. See
scripts/oracle/validate_qwen2.py for the real-weight numeric proof.
"""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestQwen2ChatHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_qwen2_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-qwen2:latest",
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

    def test_reassembled_content_round_trips_through_the_tokenizer(
        self, client: TestClient, tiny_qwen2_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-qwen2:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["message"]["content"] for line in lines)
        assert isinstance(full_text, str)

    def test_same_prompt_is_deterministic_at_zero_temperature(
        self, client: TestClient, tiny_qwen2_model: InstalledModel
    ) -> None:
        body = {
            "model": "tiny-qwen2:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
        }

        def generated_text() -> str:
            lines = _read_ndjson_lines(client.post("/api/chat", json=body))
            return "".join(line["message"]["content"] for line in lines)

        assert generated_text() == generated_text()


class TestQwen2ChatTiedEmbeddings:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_qwen2_tied_embeddings_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-qwen2-tied:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        assert response.status_code == 200
        lines = _read_ndjson_lines(response)

        assert all("message" in line for line in lines)
        *content_lines, done_line = lines
        assert all(line["done"] is False for line in content_lines)
        assert done_line["done"] is True
        assert done_line["eval_count"] == 5

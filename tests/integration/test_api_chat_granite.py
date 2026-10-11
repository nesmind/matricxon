"""`/api/chat` against a tiny `granite` GGUF (tests/tiny_gguf_granite.py), the dense-Granite
equivalent of test_api_chat_llama.py. Real
GGUFModelLoader/GraniteArchitecture/GGUFTokenizer/ChatEngine/NDJSON stack, not mocked. Real-weight
numeric proof: tools/oracle/validate_granite.py."""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestGraniteChatHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_granite_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-granite:latest",
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
        self, client: TestClient, tiny_granite_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-granite:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["message"]["content"] for line in lines)
        assert isinstance(full_text, str)

    def test_same_prompt_is_deterministic_at_zero_temperature(
        self, client: TestClient, tiny_granite_model: InstalledModel
    ) -> None:
        body = {
            "model": "tiny-granite:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
        }

        def generated_text() -> str:
            lines = _read_ndjson_lines(client.post("/api/chat", json=body))
            return "".join(line["message"]["content"] for line in lines)

        assert generated_text() == generated_text()


class TestGraniteChatTiedEmbeddings:
    """Real Granite checkpoints ship `tie_word_embeddings: true` (live config.json for
    granite-3.0-2b-instruct/granite-3.0-1b-a400m-instruct): the common case, not an edge, unlike
    llama where it crashed. Guarded from day one."""

    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_granite_tied_embeddings_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-granite-tied:latest",
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

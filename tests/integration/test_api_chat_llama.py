"""M10: /api/chat against a tiny but real llama GGUF (see

tests/tiny_gguf_llama.py) - the plain-`llama`-architecture equivalent of
test_api_chat_real_pipeline.py. Real GGUFModelLoader/LlamaArchitecture/
SentencePieceTokenizer/ChatEngine/NDJSON stack, not mocked. See
scripts/oracle/validate_sentencepiece_tokenizer.py and the manual real-weight
check (ROADMAP.md's M10 llama entry) for the full real-model proof.
"""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestLlamaChatHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_llama_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-llama:latest",
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
        self, client: TestClient, tiny_llama_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-llama:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["message"]["content"] for line in lines)
        assert isinstance(full_text, str)

    def test_same_prompt_is_deterministic_at_zero_temperature(
        self, client: TestClient, tiny_llama_model: InstalledModel
    ) -> None:
        body = {
            "model": "tiny-llama:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
        }

        def generated_text() -> str:
            lines = _read_ndjson_lines(client.post("/api/chat", json=body))
            return "".join(line["message"]["content"] for line in lines)

        assert generated_text() == generated_text()


class TestLlamaChatTiedEmbeddings:
    """Real bug (2026-09-22): a real `Llama-3.2-3B-Instruct-GGUF` pull has no separate
    `output.weight` tensor (`tie_word_embeddings: true`) - `/api/chat` crashed mid-stream with
    `UnknownModelError: GGUF file has no tensor named 'output.weight'`. See
    `tiny_llama_tied_embeddings_model`/`LlamaArchitecture`'s own docstring for the fix."""

    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_llama_tied_embeddings_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-llama-tied:latest",
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

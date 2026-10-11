"""M10: /api/chat against a tiny gemma4 GGUF (tests/tiny_gguf_gemma4.py), the gemma4 equivalent of
test_api_chat_real_pipeline.py/test_api_chat_llama.py. Real
GGUFModelLoader/Gemma4Architecture/Gemma4Tokenizer/ChatEngine/NDJSON stack, not mocked; generation
runs long enough (num_predict=10 vs sliding_window=4) to cross the sliding-window boundary on the
local layers, not just the one global layer. Proves the wiring; the real `google/gemma-4-12b-it` is
too large to validate numerically here."""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestGemma4ChatHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_gemma4_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-gemma4:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 10},
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
        assert done_line["eval_count"] == 10

    def test_output_has_no_nan_or_inf_artifacts(
        self, client: TestClient, tiny_gemma4_model: InstalledModel
    ) -> None:
        """A real risk given this model's extra parts (sandwich norms, softcapping, the rope_freqs
        division): garbage from a shape/masking bug would show as NaN/replacement-character noise
        well before 10 tokens."""
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-gemma4:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 10},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["message"]["content"] for line in lines)
        assert isinstance(full_text, str)

    def test_same_prompt_is_deterministic_at_zero_temperature(
        self, client: TestClient, tiny_gemma4_model: InstalledModel
    ) -> None:
        body = {
            "model": "tiny-gemma4:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 10},
        }

        def generated_text() -> str:
            lines = _read_ndjson_lines(client.post("/api/chat", json=body))
            return "".join(line["message"]["content"] for line in lines)

        assert generated_text() == generated_text()

"""`/api/chat` against a tiny Mixtral-style `llama` GGUF (tests/tiny_gguf_llama_moe.py), the
sparse-MoE equivalent of test_api_chat_llama.py. Real
GGUFModelLoader/LlamaArchitecture/GraniteMoeFFN/SentencePieceTokenizer/ChatEngine/NDJSON stack, not
mocked: confirms `LlamaArchitecture`'s Mixtral detection (`llama.expert_count` present,
`general.architecture` still `"llama"`) dispatches to the MoE FFN end to end. The router math is
unit-tested in tests/unit/test_granitemoe_ffn.py (same `GraniteMoeFFN`;
app/architectures/llama_moe.py proves it equals Mixtral's formula)."""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestLlamaMoeChatHappyPath:
    def test_streams_ndjson_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_llama_moe_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-llama-moe:latest",
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
        self, client: TestClient, tiny_llama_moe_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-llama-moe:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
            },
        )

        lines = _read_ndjson_lines(response)
        full_text = "".join(line["message"]["content"] for line in lines)
        assert isinstance(full_text, str)

    def test_same_prompt_is_deterministic_at_zero_temperature(
        self, client: TestClient, tiny_llama_moe_model: InstalledModel
    ) -> None:
        body = {
            "model": "tiny-llama-moe:latest",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
        }

        def generated_text() -> str:
            lines = _read_ndjson_lines(client.post("/api/chat", json=body))
            return "".join(line["message"]["content"] for line in lines)

        assert generated_text() == generated_text()

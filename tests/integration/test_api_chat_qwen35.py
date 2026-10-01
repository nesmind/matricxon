"""`/api/chat` against a tiny real `qwen35` GGUF (tests/tiny_gguf_qwen35.py): real loader,
`Qwen35Architecture`, hybrid cache, tokenizer and NDJSON stack - multi-token prefill followed by
single-token decode through both Gated DeltaNet and gated-attention layers. The recurrence math
itself is covered in tests/unit/test_qwen35_deltanet.py.
"""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel

_BODY = {
    "model": "tiny-qwen35:latest",
    "messages": [{"role": "user", "content": "hello there"}],
    "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 5},
}


def _lines(client: TestClient) -> list[dict]:
    response = client.post("/api/chat", json=_BODY)
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestQwen35Chat:
    def test_streams_chunks_ending_in_a_done_line(
        self, client: TestClient, tiny_qwen35_model: InstalledModel
    ) -> None:
        *content_lines, done_line = _lines(client)

        assert all(line["done"] is False for line in content_lines)
        assert done_line["done"] is True
        assert done_line["prompt_eval_count"] > 0
        assert done_line["eval_count"] == 5

    def test_same_prompt_is_deterministic_at_zero_temperature(
        self, client: TestClient, tiny_qwen35_model: InstalledModel
    ) -> None:
        def text() -> str:
            return "".join(line["message"]["content"] for line in _lines(client))

        assert text() == text()

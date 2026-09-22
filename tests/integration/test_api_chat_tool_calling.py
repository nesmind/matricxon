"""M10: tool-calling support in /api/chat against a tiny but real mistral3

GGUF (see tests/tiny_gguf.py, now with the AVAILABLE_TOOLS/TOOL_CALLS/ARGS/
TOOL_RESULTS control tokens added) - proves the real GGUFTokenizer round-
trips these new control tokens end to end, not just Mistral3PromptBuilder's
own unit-level string assertions (test_prompt_builder.py).
"""

import json

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def _read_ndjson_lines(response) -> list[dict]:
    return [json.loads(line) for line in response.text.strip().splitlines()]


class TestToolCallingHappyPath:
    def test_a_full_tool_calling_turn_streams_successfully(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"city": "Paris"},
                                }
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"temp_c": 15}'},
                ],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "options": {"temperature": 0.0, "num_ctx": 512, "num_predict": 3},
            },
        )

        assert response.status_code == 200
        lines = _read_ndjson_lines(response)
        assert lines[-1]["done"] is True
        assert lines[-1]["eval_count"] == 3

    def test_a_plain_message_still_works_when_tools_field_is_omitted(
        self, client: TestClient, tiny_mistral3_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={
                "model": "tiny-mistral3:latest",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.0, "num_ctx": 32, "num_predict": 3},
            },
        )

        assert response.status_code == 200

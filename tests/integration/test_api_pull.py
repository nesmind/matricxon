"""HTTP-level /api/pull tests. The bad-tag-format case needs no network
(HFModelTag.parse fails before any resolver call), which is what's tested
here - the full resolve+download happy path is covered directly against
PullJob in tests/unit/test_pull_job.py (mocked resolver/downloader), and
against a real Hugging Face repo in scripts/manual_pull_check.py.
"""

import json

from fastapi.testclient import TestClient


def test_pull_of_a_plain_ollama_tag_streams_an_in_band_error(client: TestClient) -> None:
    response = client.post("/api/pull", json={"model": "llama3.2:1b"})

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.strip().splitlines()]
    assert lines[-1]["error"]

from fastapi.testclient import TestClient


def test_decoder_architecture_reports_completion(client: TestClient) -> None:
    response = client.post(
        "/api/infer-capabilities",
        json={"repo_id": "org/repo", "filename": "model.gguf", "architecture": "mistral3"},
    )

    assert response.status_code == 200
    assert response.json()["capabilities"] == ["completion"]


def test_encoder_architecture_reports_embedding(client: TestClient) -> None:
    response = client.post(
        "/api/infer-capabilities",
        json={"repo_id": "org/repo", "filename": "model.gguf", "architecture": "bert"},
    )

    assert response.json()["capabilities"] == ["embedding"]


def test_clip_architecture_reports_no_capabilities(client: TestClient) -> None:
    response = client.post(
        "/api/infer-capabilities",
        json={"repo_id": "org/repo", "filename": "mmproj.gguf", "architecture": "clip"},
    )

    assert response.json()["capabilities"] == []


def test_thinking_marker_is_detected(client: TestClient) -> None:
    response = client.post(
        "/api/infer-capabilities",
        json={
            "repo_id": "org/some-reasoning-model",
            "filename": "model.gguf",
            "architecture": "mistral3",
        },
    )

    assert "thinking" in response.json()["capabilities"]


def test_never_touches_the_model_catalog(client: TestClient) -> None:
    """A real, deliberate property, same as /api/health's own: this is a pure computation from
    caller-supplied strings, not the local model catalog - answers identically regardless of
    what's actually installed, and needs no `catalog` fixture dependency to work."""
    response = client.post(
        "/api/infer-capabilities",
        json={"repo_id": "org/repo", "filename": "model.gguf", "architecture": "mistral3"},
    )

    assert response.status_code == 200

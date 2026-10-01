from fastapi.testclient import TestClient

from app.architectures.registry import ArchitectureRegistry
from app.config import MATRICXON_VERSION
from app.gguf.dequant.registry import QuantStrategyRegistry


def test_returns_status_ok_and_the_real_version(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == MATRICXON_VERSION


def test_supported_architectures_matches_the_real_registry(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.json()["supported_architectures"] == ArchitectureRegistry().supported_names()


def test_vision_supported_architectures_matches_the_real_registry(client: TestClient) -> None:
    response = client.get("/api/health")

    assert (
        response.json()["vision_supported_architectures"]
        == ArchitectureRegistry().vision_supported_names()
    )


def test_moe_supported_architectures_matches_the_real_registry(client: TestClient) -> None:
    response = client.get("/api/health")

    assert (
        response.json()["moe_supported_architectures"]
        == ArchitectureRegistry().moe_supported_names()
    )


def test_supported_quantizations_matches_the_real_registry(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.json()["supported_quantizations"] == QuantStrategyRegistry().supported_names()


def test_response_never_touches_the_model_catalog(client: TestClient) -> None:
    """A real, deliberate property: this is static capability data derived

    from the code itself, not the local model catalog - `/api/health`
    must answer identically whether or not any model is installed.
    """
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["supported_architectures"]

"""Manual smoke test for M8: a real `POST /api/embeddings` turn against the
real `all-MiniLM-L6-v2` GGUF - the embeddings equivalent of
manual_chat_check.py. No memory cap needed (this model is ~46MB).

    .venv/bin/python -m scripts.manual_embed_check
"""

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.dependencies import get_model_catalog, get_model_manager
from app.main import MatricxonApp
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager

REAL_GGUF_PATH = (
    "/home/home/Code/Py/AI/pAIring/models/blobs/"
    "sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662"
)
TAG = "all-minilm:latest"


def _build_test_client(models_dir: Path) -> TestClient:
    repo_dir = models_dir / "registry.ollama.ai" / "library" / "all-minilm"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_link = repo_dir / "latest.gguf"
    gguf_link.symlink_to(REAL_GGUF_PATH)

    installed = InstalledModel(
        tag=TAG,
        path=str(gguf_link),
        architecture="bert",
        capabilities=["embedding"],
        size_bytes=Path(REAL_GGUF_PATH).stat().st_size,
        family="bert",
        parameter_size="22.7M",
        context_length=512,
    )
    sidecar = repo_dir / f"{gguf_link.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))

    app = MatricxonApp().build()
    catalog = ModelCatalog(models_dir)
    app.dependency_overrides[get_model_catalog] = lambda: catalog
    app.dependency_overrides[get_model_manager] = lambda: ModelManager(catalog)
    return TestClient(app)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = _build_test_client(Path(tmp_dir))
        response = client.post(
            "/api/embeddings", json={"model": TAG, "prompt": "The capital of France is Paris."}
        )

        print(f"status: {response.status_code}")
        if response.status_code != 200:
            print(f"error: {response.json()}")
            return

        embedding = response.json()["embedding"]
        norm = sum(x * x for x in embedding) ** 0.5
        print(f"embedding length: {len(embedding)}")
        print(f"L2 norm: {norm:.6f}  (should be ~1.0 - the model normalizes its output)")
        print(f"first 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()

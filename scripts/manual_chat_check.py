"""Manual smoke test for M6: a real `POST /api/chat` turn against the full
real `ministral-3:3b` model - the HTTP-layer equivalent of M4's
manual_generate_check.py (which exercises ChatEngine directly, not the API).

Not a pass/fail check, for the same reason as manual_generate_check.py.
Uses FastAPI's TestClient - the real ASGI app, real routing, real dependency
injection, real NDJSON streaming, just without a literal TCP socket. For an
actual curl against a running server instead:

    scripts/start.sh
    curl -N -X POST http://localhost:8420/api/chat \\
        -H 'content-type: application/json' \\
        -d '{"model": "ministral-3:3b", "messages": [{"role":"user","content":"..."}]}'
    scripts/stop.sh

Run under a memory cap (same reasoning as manual_generate_check.py - the
full model needs the swap headroom added for M4):

    systemd-run --user --scope -p MemoryMax=13G -- \\
        .venv/bin/python -m scripts.manual_chat_check
"""

import argparse
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.dependencies import get_model_catalog, get_model_manager
from app.main import MatricxonApp
from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from scripts.oracle.common import DEFAULT_GGUF_PATH


def _build_test_client(gguf_path: str, models_dir: Path) -> TestClient:
    repo_dir = models_dir / "registry.ollama.ai" / "library" / "ministral-3"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_link = repo_dir / "3b.gguf"
    gguf_link.symlink_to(gguf_path)

    installed = InstalledModel(
        tag="ministral-3:3b",
        path=str(gguf_link),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=Path(gguf_path).stat().st_size,
        family="mistral3",
        parameter_size="3.8B",
        context_length=262144,
    )
    sidecar = repo_dir / f"{gguf_link.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))

    app = MatricxonApp().build()
    catalog = ModelCatalog(models_dir)
    app.dependency_overrides[get_model_catalog] = lambda: catalog
    app.dependency_overrides[get_model_manager] = lambda: ModelManager(catalog)
    return TestClient(app)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", default=str(DEFAULT_GGUF_PATH))
    parser.add_argument("--message", default="What is the capital of France?")
    parser.add_argument("--num-predict", type=int, default=30)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        client = _build_test_client(args.gguf_path, Path(tmp_dir))
        response = client.post(
            "/api/chat",
            json={
                "model": "ministral-3:3b",
                "messages": [{"role": "user", "content": args.message}],
                "options": {"temperature": 0.0, "num_predict": args.num_predict},
            },
        )

        print(f"status: {response.status_code}")
        if response.status_code != 200:
            print(f"error: {response.json()}")
            return

        full_text = ""
        telemetry = None
        for line in response.text.strip().splitlines():
            chunk = json.loads(line)
            full_text += chunk["message"]["content"]
            if chunk["done"]:
                telemetry = chunk

        print(f"message:   {args.message!r}")
        print(f"response:  {full_text!r}")
        print(f"telemetry: {telemetry}")


if __name__ == "__main__":
    main()

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_node_id, get_storage_paths
from app.main import MatricxonApp
from app.storage.paths import StoragePaths


def test_lists_every_path_with_role_size_and_free_space(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.gguf").write_bytes(b"z" * 7)
    settings = Settings(
        _env_file=None, models_dir=models, prompt_cache_dir=tmp_path / "state" / "prompt_cache"
    )
    app = MatricxonApp().build()
    app.dependency_overrides[get_storage_paths] = lambda: StoragePaths(settings, tmp_path / "p")
    app.dependency_overrides[get_node_id] = lambda: "node-1"

    body = TestClient(app).get("/api/storage").json()

    assert body["node_id"] == "node-1"
    by_name = {p["name"]: p for p in body["paths"]}
    assert by_name["models"]["role"] == "shared"
    assert by_name["models"]["size_bytes"] == 7 and by_name["models"]["exists"] is True
    assert by_name["prompt_cache"]["exists"] is False
    assert by_name["prompt_cache"]["free_bytes"] > 0
    assert by_name["cache_policy"]["role"] == "cluster_config"

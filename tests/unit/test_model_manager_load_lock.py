"""Two requests for a still-loading model share one load (ModelManager.get_or_load per-tag lock)."""

import threading
import time

from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from tests.unit.test_model_manager import FakeClock, _install_tiny_model  # noqa: F401


def test_concurrent_requests_for_one_model_load_it_once(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    _install_tiny_model(models_dir, "model-a:latest")
    manager = ModelManager(ModelCatalog(models_dir), max_loaded=2)
    real_load = manager._load
    loads: list[str] = []

    def slow_load(tag, keep_alive_seconds, now):
        loads.append(tag)
        time.sleep(0.2)  # the second request arrives mid-load
        return real_load(tag, keep_alive_seconds, now)

    monkeypatch.setattr(manager, "_load", slow_load)
    handles = []
    threads = [
        threading.Thread(target=lambda: handles.append(manager.get_or_load("model-a:latest")))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert loads == ["model-a:latest"]
    assert handles[0] is handles[1]

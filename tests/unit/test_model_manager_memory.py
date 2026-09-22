"""M10: ModelManager additions - max_loaded_models > 2 and on-the-fly dequant

- a separate file from test_model_manager.py (already at the project's
250-line cap) rather than growing it further. Covers: `max_loaded` actually
works generically past 2 (the eviction/capacity logic was never hardcoded
to 2, just defaulted to it); the new `_ensure_enough_memory_to_load` circuit
breaker that makes raising it safe on a RAM-constrained box; and that every
handle-dropping path (unload/expiry/LRU eviction) correctly releases a
never-materialized model's still-open GGUF loader (see
app/architectures/base.py's `close`) rather than leaking its mmap - the
per-architecture deferred-materialization mechanism itself is covered by
tests/unit/test_lazy_materialization.py.
"""

import json
from pathlib import Path

import pytest

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from app.server.errors import InsufficientMemoryError
from tests.tiny_gguf import build_tiny_mistral3_gguf


def _install_tiny_model(models_dir: Path, tag: str, size_bytes: int = 0) -> InstalledModel:
    repo_dir = models_dir / tag.replace(":", "_")
    repo_dir.mkdir(parents=True)
    gguf_path = repo_dir / "model.gguf"
    build_tiny_mistral3_gguf(gguf_path)

    installed = InstalledModel(
        tag=tag,
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=size_bytes or gguf_path.stat().st_size,
        family="mistral3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))
    return installed


class TestHigherCapacity:
    def test_max_loaded_three_keeps_three_models_resident(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        for tag in ("model-a:latest", "model-b:latest", "model-c:latest"):
            _install_tiny_model(models_dir, tag)
        manager = ModelManager(ModelCatalog(models_dir), max_loaded=3)

        manager.get_or_load("model-a:latest")
        manager.get_or_load("model-b:latest")
        manager.get_or_load("model-c:latest")

        tags = {handle.tag for handle in manager.list_loaded()}
        assert tags == {"model-a:latest", "model-b:latest", "model-c:latest"}

    def test_a_fourth_load_evicts_the_lru_model_under_max_loaded_three(
        self, tmp_path: Path
    ) -> None:
        models_dir = tmp_path / "models"
        for tag in ("model-a:latest", "model-b:latest", "model-c:latest", "model-d:latest"):
            _install_tiny_model(models_dir, tag)
        manager = ModelManager(ModelCatalog(models_dir), max_loaded=3)

        manager.get_or_load("model-a:latest")
        manager.get_or_load("model-b:latest")
        manager.get_or_load("model-c:latest")
        manager.get_or_load("model-d:latest")  # over max_loaded=3 - evicts model-a (LRU)

        tags = {handle.tag for handle in manager.list_loaded()}
        assert tags == {"model-b:latest", "model-c:latest", "model-d:latest"}


class TestMemoryGuard:
    def test_refuses_to_load_when_even_bf16_clearly_would_not_fit(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        models_dir = tmp_path / "models"
        # The real tiny GGUF's own tensors are far too small to ever trip this guard on their own
        # (that's the whole point of tests/tiny_gguf.py) - exact_bf16_bytes is patched here to the
        # ~10GB a real model's tensors would actually dequantize to, against a ~1GB "available",
        # so this guard is exercised the same way a real huge model would trip it.
        _install_tiny_model(models_dir, "huge:latest")
        manager = ModelManager(ModelCatalog(models_dir))
        monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: 1 * 1024**3)
        # Patched at its real definition site (app.models.load_dtype), not manager.py's own
        # imported name - the guard's input now goes through estimate_quantized_native_bytes
        # (see that function's own docstring), which calls exact_bf16_bytes via load_dtype's own
        # module namespace, not manager.py's separately-bound copy of the same name (the same
        # module-split import-binding gotcha this project already tracks elsewhere).
        monkeypatch.setattr(
            "app.models.load_dtype.exact_bf16_bytes", lambda tensor_infos: 10 * 1024**3
        )

        with pytest.raises(InsufficientMemoryError):
            manager.get_or_load("huge:latest")

    def test_error_message_names_the_offending_tag_and_currently_loaded_models(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "already-loaded:latest")
        _install_tiny_model(models_dir, "huge:latest")
        manager = ModelManager(ModelCatalog(models_dir), max_loaded=99)
        monkeypatch.setattr(
            "app.models.memory_guard.available_memory_bytes", lambda: 100 * 1024**3
        )
        # See test_refuses_to_load_when_even_bf16_clearly_would_not_fit's own comment - patched for
        # this whole test (not just the second load) so "already-loaded" loading successfully
        # against 100GB "available" also proves the guard's math, not just that it was skipped.
        monkeypatch.setattr(
            "app.models.load_dtype.exact_bf16_bytes", lambda tensor_infos: 10 * 1024**3
        )
        manager.get_or_load("already-loaded:latest")
        monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: 1 * 1024**3)

        with pytest.raises(InsufficientMemoryError) as exc_info:
            manager.get_or_load("huge:latest")

        message = str(exc_info.value)
        assert "huge:latest" in message
        assert "already-loaded:latest" in message

    def test_proceeds_when_memory_clearly_is_sufficient(self, tmp_path: Path, monkeypatch) -> None:
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest")
        manager = ModelManager(ModelCatalog(models_dir))
        monkeypatch.setattr(
            "app.models.memory_guard.available_memory_bytes", lambda: 100 * 1024**3
        )

        handle = manager.get_or_load("model-a:latest")

        assert handle.tag == "model-a:latest"

    def test_a_lower_configured_safety_margin_allows_a_load_the_default_would_refuse(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """`memory_safety_margin` (Settings.memory_safety_margin / MATRICXON_MEMORY_SAFETY_MARGIN -
        see app/config.py) flows into ModelManager's own constructor, not just load_dtype's module
        default - a 5GB model against 6GB available needs 7.5GB at the 1.5x default (refused) but
        only 5.5GB at 1.1x (allowed)."""
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest")
        manager = ModelManager(ModelCatalog(models_dir), memory_safety_margin=1.1)
        monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: 6 * 1024**3)
        monkeypatch.setattr(
            "app.models.load_dtype.exact_bf16_bytes", lambda tensor_infos: 5 * 1024**3
        )

        handle = manager.get_or_load("model-a:latest")

        assert handle.tag == "model-a:latest"

    def test_a_higher_configured_safety_margin_refuses_a_load_the_default_would_allow(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest")
        manager = ModelManager(ModelCatalog(models_dir), memory_safety_margin=1.8)
        # 5GB * 1.5 (default) = 7.5GB <= 8GB available (would proceed by default), but
        # 5GB * 1.8 (configured) = 9GB > 8GB available.
        monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: 8 * 1024**3)
        monkeypatch.setattr(
            "app.models.load_dtype.exact_bf16_bytes", lambda tensor_infos: 5 * 1024**3
        )

        with pytest.raises(InsufficientMemoryError) as exc_info:
            manager.get_or_load("model-a:latest")

        assert "1.8x safety margin" in str(exc_info.value)

    def test_proceeds_when_available_memory_is_unknown(self, tmp_path: Path, monkeypatch) -> None:
        """Same "assume it fits" policy `select_load_dtype` already uses when

        /proc/meminfo isn't available (e.g. non-Linux) - an unanswerable
        check must never block every load outright.
        """
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest", size_bytes=10 * 1024**3)
        manager = ModelManager(ModelCatalog(models_dir))
        monkeypatch.setattr("app.models.memory_guard.available_memory_bytes", lambda: None)

        handle = manager.get_or_load("model-a:latest")

        assert handle.tag == "model-a:latest"


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class TestEvictionReleasesUnmaterializedLoaders:
    """Every path that drops a handle must call `architecture.close()` - a

    model evicted before its first real forward pass still has a live GGUF
    mmap it never got to release itself (see app/architectures/base.py's
    `close`). Each handle here is deliberately never used for a forward pass
    (`is_materialized` stays False throughout), so a passing test proves the
    eviction path itself does the releasing, not a forward call.
    """

    def test_unload_releases_the_loader(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest")
        manager = ModelManager(ModelCatalog(models_dir))
        handle = manager.get_or_load("model-a:latest")
        assert handle.architecture.is_materialized is False

        manager.unload("model-a:latest")

        assert handle.architecture._pending_loader is None

    def test_lru_eviction_releases_the_loader(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest")
        _install_tiny_model(models_dir, "model-b:latest")
        manager = ModelManager(ModelCatalog(models_dir), max_loaded=1)
        handle_a = manager.get_or_load("model-a:latest")
        assert handle_a.architecture.is_materialized is False

        manager.get_or_load("model-b:latest")  # over max_loaded=1 - evicts model-a (LRU)

        assert handle_a.architecture._pending_loader is None

    def test_expiry_eviction_releases_the_loader(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        _install_tiny_model(models_dir, "model-a:latest")
        clock = _FakeClock()
        manager = ModelManager(ModelCatalog(models_dir), default_keep_alive_seconds=10, clock=clock)
        handle = manager.get_or_load("model-a:latest")
        assert handle.architecture.is_materialized is False

        clock.advance(11)
        manager.evict_expired()

        assert handle.architecture._pending_loader is None

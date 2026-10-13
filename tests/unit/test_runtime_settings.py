"""Settings changed without a restart: loaded models take the new limits at once, the load-time
choices apply from the next load, and the endpoint is an admin route."""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.dependencies import get_model_manager
from app.main import MatricxonApp
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.runtime.prompt_cache import PromptCache
from app.schemas.runtime_settings import RuntimeSettingsUpdate
from app.server.runtime_settings import RuntimeSettingsApplier
from tests.unit.test_model_manager import _install_tiny_model
from tests.unit.test_prompt_cache_pool import CONV_A, CONV_B, _generate, _pool, llama  # noqa: F401


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    models_dir = tmp_path / "models"
    for tag in ("model-a:latest", "model-b:latest", "model-c:latest"):
        _install_tiny_model(models_dir, tag)
    return ModelManager(ModelCatalog(models_dir), max_loaded=2, default_keep_alive_seconds=300)


class FakeManager:
    def __init__(self) -> None:
        self.limits: dict[str, object] = {}

    def set_limits(self, **limits: object) -> None:
        self.limits.update(limits)


class TestModelManagerSetLimits:
    def test_loaded_model_takes_new_limits_now(self, manager: ModelManager) -> None:
        handle = manager.get_or_load("model-a:latest")
        manager.set_limits(
            default_keep_alive_seconds=900,
            prompt_cache_slots=2,
            prompt_cache_budget_mb=64,
            max_decode_batch=3,
        )
        assert handle.keep_alive_seconds == 900
        assert handle.prompt_cache.budget_bytes == 64 << 20
        assert handle.prompt_cache._max_slots == 2
        assert handle.worker._max_active == 3
        manager.shutdown()

    def test_higher_max_loaded_keeps_more_models(self, manager: ModelManager) -> None:
        manager.set_limits(max_loaded=3)
        for tag in ("model-a:latest", "model-b:latest", "model-c:latest"):
            manager.get_or_load(tag)
        assert len(manager.list_loaded()) == 3
        manager.shutdown()

    def test_none_keeps_a_value(self, manager: ModelManager) -> None:
        manager.set_limits(max_decode_batch=5)
        manager.set_limits(max_loaded=4)
        assert manager._max_decode_batch == 5 and manager._max_loaded == 4

    def test_load_time_choices_are_stored_for_the_next_load(self, manager: ModelManager) -> None:
        manager.set_limits(enable_quantized_native_compute=True, gpu_weight_mode="packed")
        assert manager._enable_quantized_native_compute is True
        assert manager._gpu_weight_mode == "packed"


def test_prompt_cache_set_limits_evicts_down_to_the_new_count(llama) -> None:  # noqa: F811
    cache: PromptCache = _pool()
    _generate(llama, cache, CONV_A)
    _generate(llama, cache, CONV_B)
    assert cache.slot_count == 2
    cache.set_limits(1, cache.budget_bytes)
    assert cache.slot_count == 1


class TestApplier:
    def test_sent_values_reach_settings_and_manager(self) -> None:
        own, fake = Settings(_env_file=None), FakeManager()
        result = RuntimeSettingsApplier(fake, own).apply(
            RuntimeSettingsUpdate(max_loaded_models=3, gpu_weight_mode="packed")
        )
        assert own.max_loaded_models == 3 and own.gpu_weight_mode == "packed"
        assert fake.limits == {"max_loaded": 3, "gpu_weight_mode": "packed"}
        assert result.applied_now == ["max_loaded_models"]
        assert result.applied_next_load == ["gpu_weight_mode"]

    def test_null_goes_back_to_the_default(self) -> None:
        own, fake = Settings(_env_file=None, prompt_cache_slots=9), FakeManager()
        RuntimeSettingsApplier(fake, own).apply(RuntimeSettingsUpdate(prompt_cache_slots=None))
        assert own.prompt_cache_slots == Settings.model_fields["prompt_cache_slots"].default
        assert fake.limits == {"prompt_cache_slots": own.prompt_cache_slots}

    def test_fields_left_out_are_untouched(self) -> None:
        own, fake = Settings(_env_file=None, max_decode_batch=6), FakeManager()
        result = RuntimeSettingsApplier(fake, own).apply(RuntimeSettingsUpdate())
        assert own.max_decode_batch == 6 and fake.limits == {}
        assert result.applied_now == [] and result.applied_next_load == []

    def test_log_level_changes_the_root_logger(self) -> None:
        root = logging.getLogger()
        before = root.level
        try:
            RuntimeSettingsApplier(FakeManager(), Settings(_env_file=None)).apply(
                RuntimeSettingsUpdate(log_level=2)
            )
            assert root.level == logging.DEBUG
        finally:
            root.setLevel(before)


class TestEndpoint:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, FakeManager]:
        # The endpoint writes to the global settings; monkeypatch puts them back.
        monkeypatch.setattr(settings, "max_decode_batch", settings.max_decode_batch)
        monkeypatch.setattr(settings, "admin_secret", None)  # this machine's .env may set one
        fake = FakeManager()
        app = MatricxonApp().build()
        app.dependency_overrides[get_model_manager] = lambda: fake
        return TestClient(app), fake

    def test_applies_a_change_without_a_secret(self, client) -> None:
        test_client, fake = client
        response = test_client.put("/api/runtime-settings", json={"max_decode_batch": 4})
        assert response.status_code == 200
        assert response.json() == {"applied_now": ["max_decode_batch"], "applied_next_load": []}
        assert fake.limits == {"max_decode_batch": 4}

    def test_rejects_an_unsigned_call_when_a_secret_is_set(self, client, monkeypatch) -> None:
        test_client, fake = client
        monkeypatch.setattr(settings, "admin_secret", "s3cret")
        response = test_client.put("/api/runtime-settings", json={"max_decode_batch": 4})
        assert response.status_code == 401 and fake.limits == {}

    def test_rejects_out_of_range_values(self, client) -> None:
        test_client, _ = client
        response = test_client.put("/api/runtime-settings", json={"max_decode_batch": 0})
        assert response.status_code == 422

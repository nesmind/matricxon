"""Applies setting changes to a running server, so most Settings need no restart."""

import logging

from app.config import Settings
from app.models.manager import ModelManager
from app.native.fused_ops import FusedOps
from app.native.gemm import NativeGemm
from app.schemas.runtime_settings import RuntimeSettingsResult, RuntimeSettingsUpdate
from app.server.log_filters import LOG_LEVELS_BY_SETTING

logger = logging.getLogger(__name__)

# Settings name -> ModelManager.set_limits keyword.
_MANAGER_NOW = {
    "max_loaded_models": "max_loaded",
    "default_keep_alive_seconds": "default_keep_alive_seconds",
    "memory_safety_margin": "memory_safety_margin",
    "prompt_cache_slots": "prompt_cache_slots",
    "prompt_cache_budget_mb": "prompt_cache_budget_mb",
    "max_decode_batch": "max_decode_batch",
}
_MANAGER_NEXT_LOAD = {
    "enable_quantized_native_compute": "enable_quantized_native_compute",
    "gpu_weight_mode": "gpu_weight_mode",
}


class RuntimeSettingsApplier:
    """Writes each changed value to `settings` (so routers read it) and to the live objects."""

    def __init__(self, manager: ModelManager, settings: Settings) -> None:
        self._manager = manager
        self._settings = settings

    def _values(self, change: RuntimeSettingsUpdate) -> dict[str, object]:
        """Fields sent in the request; a null means the default."""
        values = {}
        for name in change.model_fields_set:
            value = getattr(change, name)
            values[name] = Settings.model_fields[name].default if value is None else value
        return values

    def apply(self, change: RuntimeSettingsUpdate) -> RuntimeSettingsResult:
        values = self._values(change)
        for name, value in values.items():
            setattr(self._settings, name, value)

        limits = {_MANAGER_NOW[n]: v for n, v in values.items() if n in _MANAGER_NOW}
        limits |= {_MANAGER_NEXT_LOAD[n]: v for n, v in values.items() if n in _MANAGER_NEXT_LOAD}
        if limits:
            self._manager.set_limits(**limits)
        if "log_level" in values:
            logging.getLogger().setLevel(LOG_LEVELS_BY_SETTING[self._settings.log_level])
        if "gemv_backend" in values:
            # Models load their kernels at load time, so loaded ones keep theirs.
            NativeGemm.configure(self._settings.gemv_backend, self._settings.torch_threads)
            FusedOps.configure(self._settings.enable_fused_ops)

        next_load = {*_MANAGER_NEXT_LOAD, "gemv_backend"}
        result = RuntimeSettingsResult(
            applied_now=sorted(n for n in values if n not in next_load),
            applied_next_load=sorted(n for n in values if n in next_load),
        )
        logger.info("settings changed without restart: %s", ", ".join(sorted(values)) or "none")
        return result

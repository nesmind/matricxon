"""Settings that change without a restart (PUT /api/runtime-settings). A field left out stays as it
is; a field sent as null goes back to its default (app.config.Settings)."""

from typing import Literal

from pydantic import BaseModel, Field


class RuntimeSettingsUpdate(BaseModel):
    # Apply now, also to loaded models.
    max_loaded_models: int | None = Field(default=None, ge=1, le=64)
    default_keep_alive_seconds: int | None = Field(default=None, ge=60, le=86_400)
    memory_safety_margin: float | None = Field(default=None, ge=1.1, le=1.8)
    prompt_cache_slots: int | None = Field(default=None, ge=1, le=64)
    prompt_cache_budget_mb: int | None = Field(default=None, ge=64, le=1_048_576)
    max_decode_batch: int | None = Field(default=None, ge=1, le=64)
    log_level: int | None = Field(default=None, ge=0, le=2)
    # Apply from the next model load.
    enable_quantized_native_compute: bool | None = None
    gemv_backend: Literal["numba", "native"] | None = None
    gpu_weight_mode: Literal["dequantized", "packed"] | None = None


class RuntimeSettingsResult(BaseModel):
    applied_now: list[str]
    applied_next_load: list[str]

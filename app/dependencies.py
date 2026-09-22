from functools import lru_cache

from app.config import settings
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager


@lru_cache
def get_model_catalog() -> ModelCatalog:
    return ModelCatalog(settings.ensure_models_dir())


@lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager(
        get_model_catalog(),
        max_loaded=settings.max_loaded_models,
        default_keep_alive_seconds=settings.default_keep_alive_seconds,
        memory_safety_margin=settings.memory_safety_margin,
        enable_mixed_precision_loading=settings.enable_mixed_precision_loading,
        enable_quantized_native_compute=settings.enable_quantized_native_compute,
    )

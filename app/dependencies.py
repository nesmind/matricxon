from functools import lru_cache

from app.config import settings
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.runtime.cache_crypto import CacheKeyStore
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore


@lru_cache
def get_model_catalog() -> ModelCatalog:
    return ModelCatalog(settings.ensure_models_dir())


@lru_cache
def get_cache_store() -> PersistentCacheStore:
    state_dir = settings.prompt_cache_dir.parent
    policy = PersistencePolicy.load(
        state_dir / "prompt_cache_policy.json",
        settings.persist_prompt_cache,
        settings.persist_cache_budget_mb,
        settings.persist_cache_ttl_hours,
    )
    cipher = CacheKeyStore(state_dir / "prompt_cache.key", settings.cache_key).load()
    return PersistentCacheStore(settings.prompt_cache_dir, cipher, policy)


@lru_cache
def get_model_manager() -> ModelManager:
    return ModelManager(
        get_model_catalog(),
        cache_store=get_cache_store(),
        max_loaded=settings.max_loaded_models,
        default_keep_alive_seconds=settings.default_keep_alive_seconds,
        memory_safety_margin=settings.memory_safety_margin,
        enable_mixed_precision_loading=settings.enable_mixed_precision_loading,
        enable_quantized_native_compute=settings.enable_quantized_native_compute,
        prompt_cache_slots=settings.prompt_cache_slots,
        prompt_cache_budget_mb=settings.prompt_cache_budget_mb,
        max_decode_batch=settings.max_decode_batch,
        device=settings.device,
        gpu_weight_mode=settings.gpu_weight_mode,
    )

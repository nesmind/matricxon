import os
from functools import lru_cache

from app.cluster.configure import NodeConfigurator
from app.cluster.freshness import FreshnessChecker
from app.cluster.provisioning import EnvFile, NodeProvisioner, restart_in_place
from app.cluster.readiness import ClusterReadiness
from app.cluster.reset import NodeResetter
from app.cluster.shared_settings import SharedSettings
from app.config import settings
from app.consensus.factory import ConsensusFactory
from app.consensus.runtime import ConsensusRuntime
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.runtime.cache_crypto import CacheKeyStore
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.storage.node_identity import NodeIdentity
from app.storage.paths import StoragePaths


@lru_cache
def get_model_catalog() -> ModelCatalog:
    return ModelCatalog(settings.ensure_models_dir())


@lru_cache
def get_storage_paths() -> StoragePaths:
    return StoragePaths(settings)


@lru_cache
def get_node_id() -> str:
    return NodeIdentity(get_storage_paths().node_id_file).load()


@lru_cache
def get_consensus_runtime() -> ConsensusRuntime | None:
    paths = get_storage_paths()
    factory = ConsensusFactory(
        settings, get_node_id(), paths.consensus_journal, paths.cache_key_file
    )
    return factory.build()


@lru_cache
def get_cache_store() -> PersistentCacheStore:
    state_dir = settings.prompt_cache_dir.parent
    policy = PersistencePolicy.load(
        state_dir / "prompt_cache_policy.json",
        settings.persist_prompt_cache,
        settings.persist_cache_budget_mb,
        settings.persist_cache_ttl_hours,
    )
    cipher = CacheKeyStore(
        state_dir / "prompt_cache.key", settings.cache_key, settings.cluster_role == "off"
    ).load()
    store = PersistentCacheStore(settings.prompt_cache_dir, cipher, policy, get_node_id())
    runtime = get_consensus_runtime()
    if runtime is not None and runtime.keeper is not None:
        store.may_clean = lambda: runtime.keeper.holds("cache_clean")
        runtime.keeper.add_task("cache_clean", store.enforce_limits)
    return store


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
        prompt_cache_context_shift=settings.prompt_cache_context_shift,
        max_decode_batch=settings.max_decode_batch,
        device=settings.device,
        gpu_weight_mode=settings.gpu_weight_mode,
    )


@lru_cache
def get_cluster_readiness() -> ClusterReadiness:
    return ClusterReadiness(
        settings,
        get_storage_paths(),
        get_node_id(),
        consensus_running=lambda: get_consensus_runtime() is not None,
    )


def get_freshness_checker() -> FreshnessChecker:
    return FreshnessChecker(settings, get_storage_paths())


def _env_file() -> EnvFile:
    """The.env file, with the real environment kept in step (EnvFile)."""
    return EnvFile(get_storage_paths().env_file, os.environ)


def get_node_provisioner() -> NodeProvisioner:
    return NodeProvisioner(get_freshness_checker(), _env_file())


def get_node_resetter() -> NodeResetter:
    runtime = get_consensus_runtime()
    return NodeResetter(
        settings,
        get_storage_paths(),
        _env_file(),
        lambda: runtime is not None and runtime.node.is_voter,
        _restart_gracefully,
    )


@lru_cache
def get_shared_settings() -> SharedSettings | None:
    runtime = get_consensus_runtime()
    if runtime is None:
        return None
    env = _env_file()
    return SharedSettings(runtime, settings, env, _restart_gracefully, cache=get_cache_store)


def _restart_gracefully() -> None:
    """Models unloaded and chat caches saved first (as a server stop), then the process replaces
    itself with a fresh one reading the new settings."""
    get_model_manager().shutdown()
    restart_in_place()


def get_node_configurator() -> NodeConfigurator:
    paths = get_storage_paths()
    return NodeConfigurator(settings, paths, _env_file(), _restart_gracefully)

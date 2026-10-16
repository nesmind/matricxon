from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.load_dtype import MEMORY_SAFETY_MARGIN

# A software version, not an env-overridable knob; GET /api/version reads it.
MATRICXON_VERSION = "0.1.0"


class Settings(BaseSettings):
    # A `.env` here survives restarts (start.sh, systemd, pAIring); a real environment variable
    # still wins.
    model_config = SettingsConfigDict(env_prefix="MATRICXON_", env_file=".env")

    host: str = "0.0.0.0"
    port: int = 8420

    models_dir: Path = Path("./data/models")

    default_keep_alive_seconds: int = 300
    # Conversation caches (PromptCache) kept per loaded model, and their total memory (MB, caches +
    # recurrent snapshots). The first limit hit evicts the least recently used one; one is always
    # kept.
    prompt_cache_slots: int = Field(default=4, ge=1, le=64)
    prompt_cache_budget_mb: int = Field(default=2048, ge=64)
    # When pAIring trims a chat's oldest messages, drop those tokens from the cache and keep the
    # rest
    # (PromptCache._shift_trimmed_chat). Slightly approximate; turn off to always re-read exactly.
    prompt_cache_context_shift: bool = True
    # Encrypted disk tier below the RAM pool (PersistentCacheStore): evicted or still-cached
    # conversations survive unload/stop. First-run defaults; admins change them at runtime
    # (GET/PUT /api/cache/persistence), saved in `prompt_cache_policy.json` beside the cache dir.
    # The AES
    # key is generated into `prompt_cache.key` there (0600) or taken from MATRICXON_CACHE_KEY.
    persist_prompt_cache: bool = True
    persist_cache_budget_mb: int = Field(default=4096, ge=0)
    persist_cache_ttl_hours: int = Field(default=168, ge=1)
    prompt_cache_dir: Path = Path("./data/prompt_cache")
    cache_key: str | None = None
    # Signs the admin routes (cluster reset/configure/provision, voters, pull/delete/copy/create,
    # cache policy): pAIring holds the same secret and signs its calls. Unset = those routes stay
    # open (protect the network instead).
    admin_secret: str | None = None
    # "voter" = consensus only: never loads or serves a model, so pAIring routes no chats to it.
    node_kind: Literal["compute", "voter"] = "compute"
    # Per-node state (node id); never shared, unlike the cache dir.
    node_state_dir: Path = Path("./data")
    # "off" (standalone), "seed" (the node a cluster grows from) or "joiner" (a fresh install added
    # to
    # it). A cluster role never creates a cache key (all nodes share one).
    cluster_role: Literal["off", "seed", "joiner"] = "off"
    # Consensus (docs/CLUSTER_PAXOS.md); off unless `consensus_genesis` is set: the first voters as
    # "id=url,id=url" (same list on every node). `consensus_node_id` defaults to the persisted node
    # id;
    # `consensus_secret` signs messages between nodes (default: the shared cache key).
    consensus_genesis: str = ""
    consensus_node_id: str | None = None
    consensus_secret: str | None = None
    # How other nodes reach this one (e.g. "http://10.0.0.5:8420"). With a cluster role it makes
    # this node
    # a voter (a seed starts the group; a joiner is given `consensus_seeds`).
    advertise_url: str | None = None
    consensus_seeds: str = ""
    # Voters the leader maintains: promotes caught-up learners and replaces a lost voter
    # (docs/CLUSTER_PAXOS.md). 0 = never changes voters by itself.
    consensus_target_voters: int = Field(default=3, ge=0, le=9)
    # Applied slots between log snapshots; a node keeps one to two intervals of log.
    consensus_snapshot_every: int = Field(default=5000, ge=10)
    # Leader heartbeat interval, and the silence before a node campaigns (one to two times the
    # election
    # time). Faster failover costs traffic.
    consensus_heartbeat_seconds: float = Field(default=0.2, gt=0.01, le=5)
    consensus_election_seconds: float = Field(default=1.0, gt=0.1, le=60)
    # Compute nodes wait this many times longer before campaigning, so a voter-only node normally
    # leads (it is not busy computing). 1 = no preference.
    consensus_compute_patience: float = Field(default=1.0, ge=1, le=10)
    log_consensus_traffic: bool = False
    # Skip the shared-filesystem-type check (single-host test labs, mounts the probe can't
    # classify).
    cluster_trust_shared_paths: bool = False
    # Replies running at once on one loaded model, and the most batched into one decode forward pass
    # (ModelWorker / BatchDecoder). More users wait in line. Batching pays off with the native C
    # kernels.
    max_decode_batch: int = Field(default=8, ge=1, le=64)
    # 2 matches pAIring's usage. Raising it (MATRICXON_MAX_LOADED_MODELS) is safe with enough RAM: a
    # load that
    # clearly won't fit fails with InsufficientMemoryError
    # (ModelManager._ensure_enough_memory_to_load).
    max_loaded_models: int = 2
    # Headroom required beyond a model's exact bf16 weight size before a load
    # (ModelManager._ensure_enough_memory_to_load/select_load_dtype). Default 1.2x. Bounded 1.1-1.8:
    # lower
    # leaves too little for activations/KV cache, higher rejects loads that would likely work. Same
    # range as
    # pAIring's Settings > External servers > Matricxon form (MATRICXON_MEMORY_SAFETY_MARGIN).
    memory_safety_margin: float = Field(default=MEMORY_SAFETY_MARGIN, ge=1.1, le=1.8)
    # Off by default. Mixed per-layer float32/bf16 loading (mistral3 only, `ModelManager._load`) was
    # meant
    # to speed up CPUs without hardware bf16, but measured slower (~26s/token vs ~9.2s/token uniform
    # bf16 on
    # a 15GB machine): the float32 budget left no headroom for KV cache, activations or the mmap
    # page
    # cache, and memory pressure dominated. Kept (unit-tested: `load_dtype.group_bytes_by_layer`/
    # `plan_layer_dtypes`); a budget with headroom may still win.
    enable_mixed_precision_loading: bool = False
    # On by default. A permanent choice between two load strategies: off dequantizes every weight to
    # bf16/float32 once and keeps it in RAM (Ministral-3B Q4_K_M ~7.5GB vs ~2.6GB elsewhere; see
    # estimate_ram_gb); on uses `QuantizedLinear` for tensors of the packed types in
    # `quantized_gemv_registry.GEMV_KERNELS`, with no full dequantized weight, at ~1.9-4x slower
    # decode per
    # kernel (see quantized_gemv.py). With the C kernels, Qwen3.5-4B Q4_K_M decodes ~10x faster than
    # Numba
    # and fits a 15GB machine. The old path (MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE=false) stays.
    enable_quantized_native_compute: bool = True
    # Kernels for `QuantizedLinear` (needs enable_quantized_native_compute): "numba" (float GEMV) or
    # "native" (default), the C integer kernels in app/native/, built on first use; they also handle
    # prefill for those tensors. Falls back to numba per unsupported type/shape, or entirely without
    # a C
    # compiler.
    gemv_backend: Literal["numba", "native"] = "native"

    # One native call instead of several torch dispatches for per-layer RMSNorm, rotary embedding,
    # decode
    # attention and the sampler (app/native/fused_ops.py); C kernels with gemv_backend "native",
    # Numba
    # otherwise. Seeded sampling stays reproducible but draws a different stream than the torch
    # path.
    enable_fused_ops: bool = True

    # "cpu" (default), "cuda"/"cuda:N" or "mps" (Apple Silicon): EXPERIMENTAL GPU mode (dense
    # architectures marked `SUPPORTS_GPU`, never quantized-native). A missing GPU fails the first
    # load instead of using the CPU.
    device: str = "cpu"
    # With a GPU `device`: "dequantized" (default) keeps weights in bf16/fp16 (fastest, ~2 bytes of
    # VRAM per
    # parameter); "packed" keeps supported quant types (Q4_0/4_1/5_0/5_1/8_0, Q2_K-Q6_K, Q8_K,
    # IQ4_NL/IQ4_XS)
    # quantized in VRAM and dequantizes per call: ~3-4x less VRAM for 4-bit, slower. Other types are
    # dequantized.
    gpu_weight_mode: Literal["dequantized", "packed"] = "dequantized"
    # None = every CPU core; PyTorch's default used only 2 of 4 cores (applied in
    # app.main.MatricxonApp).
    # Ignored when `device` isn't "cpu".
    torch_threads: int | None = None

    # 0 (default): warnings/errors only. 1: per-request milestones (model load, prompt build, each
    # token,
    # summary). 2: also a line per decoder layer per forward pass. Mapped to `logging` levels in
    # app.main.MatricxonApp; the level-2 trace is in each architecture's `_forward_impl`.
    log_level: int = Field(default=0, ge=0, le=2)

    def ensure_models_dir(self) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir


settings = Settings()

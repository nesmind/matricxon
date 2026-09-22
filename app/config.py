from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.load_dtype import MEMORY_SAFETY_MARGIN

# Not env-overridable (unlike Settings below) - a real software version, not a
# runtime knob. No other source of truth exists yet (pyproject.toml carries no
# [project] version table); GET /api/version reads this directly.
MATRICXON_VERSION = "0.1.0"


class Settings(BaseSettings):
    # env_file: a real env var set on the shell that launches this process (e.g. via
    # `scripts/start.sh`) only lasts until whatever restarts the process next - confirmed live
    # (2026-09-21): a restart outside that one shell invocation lost MATRICXON_LOG_LEVEL entirely,
    # silently reverting to level 0. A `.env` file next to this one survives any restart
    # regardless of what triggers it (scripts/start.sh, systemd, pAIring's own process
    # management, ...), since pydantic-settings reads it on every `Settings()` construction - a
    # real environment variable, when one is set, still takes priority over this file (pydantic-
    # settings' own documented precedence), so this is additive, not a behavior change for anyone
    # already setting env vars explicitly.
    model_config = SettingsConfigDict(env_prefix="MATRICXON_", env_file=".env")

    host: str = "0.0.0.0"
    port: int = 8420

    models_dir: Path = Path("./data/models")

    default_keep_alive_seconds: int = 300
    # 2 (not 1) matches pAIring's real usage - see ModelManager's own docstring. Safe to raise via
    # MATRICXON_MAX_LOADED_MODELS on a machine with enough RAM: ModelManager's eviction/capacity
    # logic is generic, not hardcoded to 2, and a real load that clearly wouldn't fit now fails
    # closed with InsufficientMemoryError instead of risking an OS-level OOM kill (see
    # ModelManager._ensure_enough_memory_to_load).
    max_loaded_models: int = 1
    # Headroom `ModelManager._ensure_enough_memory_to_load`/`select_load_dtype` require beyond a
    # model's own exact bf16 weight size (see `load_dtype.exact_bf16_bytes`) before allowing a
    # load - MEMORY_SAFETY_MARGIN's own default (1.5x). Bounded 1.1-1.8: below 1.1 leaves too
    # little room for activations/KV-cache to mean anything as a *safety* margin, above 1.8
    # rejects loads a real machine could likely still handle (see pAIring's Settings > External
    # servers > Matricxon form, which exposes this same 1.1-1.8 range as
    # MATRICXON_MEMORY_SAFETY_MARGIN).
    memory_safety_margin: float = Field(default=MEMORY_SAFETY_MARGIN, ge=1.1, le=1.8)
    # Off by default. `ModelManager._load`'s mixed per-layer float32/bf16 loading (mistral3 only)
    # was built to speed up generation on a CPU with no hardware bf16 acceleration by running as
    # many decoder layers in float32 as the memory budget allows, bf16 for the rest, instead of the
    # whole model paying PyTorch's ~60x-slower emulated-bf16 path uniformly. Real end-to-end
    # verification (2026-09-20, this exact 15GB-RAM machine, real ministral-3:3b weights) found the
    # opposite of the intended effect: ~26s/token versus a ~9.2s/token uniform-bf16 control run
    # under matched conditions - not swap contamination (the control, run back-to-back on the same
    # degraded-swap state, reproduced the documented baseline almost exactly). Budgeting float32
    # layers against `available_memory_bytes()` alone, with no headroom reserved for the KV cache,
    # activations, or the source GGUF's own mmap'd page cache, pushes this machine's resident
    # footprint close enough to its real 15GB ceiling that memory-pressure overhead dwarfs the
    # intended per-layer compute win - real, not a plausible-sounding just-so story: the mixed run
    # spent 32% of wall-clock in kernel time and had real idle/wait time, the control run had
    # neither. Left disabled here (real, tested, correct as pure functions -
    # `load_dtype.group_bytes_by_layer`/`plan_layer_dtypes`, both still unit-tested against the
    # algorithm directly) rather than deleted, since a properly-headroomed budget might still yield
    # a real win - that redesign is real, needed follow-up work, not done yet.
    enable_mixed_precision_loading: bool = False
    # Off by default. Real, permanent, user-selectable choice between two load strategies -
    # not a temporary rollout toggle (explicit user direction, 2026-09-21): today's path
    # (`ModelManager._load` -> `_materialize_weights`, every architecture) dequantizes every
    # weight to bf16/float32 *once* on first use, then holds all of it in RAM for the model's
    # entire loaded lifetime - real, confirmed gap this exposes (see estimate_ram_gb's own
    # docstring): a real Ministral-3B Q4_K_M file needs ~7.5GB on Matricxon this way, versus
    # ~2.6GB for the identical file elsewhere. When True, `QuantizedLinear`
    # (app/architectures/quantized_linear.py) is used instead for a tensor whose real GGUF type
    # is one of the 11 packed types `app.gguf.dequant.quantized_gemv_registry.GEMV_KERNELS`
    # covers - real quantized-native compute, never materializing a full dequantized weight, at
    # the real, honest cost of decode running ~1.9-4x slower per real kernel (see
    # quantized_gemv.py's own docstring for why, and the real redesign attempts already tried).
    # Defaults off for the same reason `enable_mixed_precision_loading` above does: ship behind a
    # flag, let real measurement (not optimism) decide whether this becomes the default -  the
    # old path itself is never removed either way.
    enable_quantized_native_compute: bool = False

    device: str = "cpu"
    # None means "use every CPU core" (os.cpu_count()) - PyTorch's own default heuristic measured
    # live as only 2 threads on a real 4-core machine, silently leaving half the CPU idle during
    # every matmul in the hot generation path (see app.main.MatricxonApp's own docstring on where
    # this gets applied). Irrelevant when `device` isn't "cpu".
    torch_threads: int | None = None

    # 0 (default): warnings/errors only, matching matricxon's historical behavior (no logging
    # existed at all before this). 1: coarse per-request pipeline milestones (model load, prompt
    # build, each generated token, generation summary). 2: adds a line per real decoder-layer
    # iteration for every forward pass (prefill and each decode step alike) - see
    # app.main.MatricxonApp's own docstring for where this maps onto Python's stdlib `logging`
    # levels, and each architecture's own `_forward_impl` for where the level-2 per-layer trace
    # actually lives.
    log_level: int = Field(default=0, ge=0, le=2)

    def ensure_models_dir(self) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self.models_dir


settings = Settings()

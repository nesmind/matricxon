from collections.abc import Callable

from app.models.load_dtype import available_memory_bytes
from app.server.errors import InsufficientMemoryError


def ensure_enough_memory_to_load(
    tag: str,
    bf16_bytes: int,
    safety_margin: float,
    loaded_tags: list[str],
    available_fn: Callable[[], int | None] | None = None,
    memory_kind: str = "memory",
) -> None:
    """A hard circuit breaker in front of a model load, not just a dtype choice: `select_load_dtype`
    picks the smallest footprint (bf16) when float32 wouldn't fit but never refuses, so on a tight
    machine (e.g. `max_loaded_models` raised past pAIring's default 2) the load would risk an OS OOM
    kill instead of one request failing cleanly. Called from `ModelManager._load` after eviction in
    `get_or_load`, so `available_memory_bytes` reflects what eviction freed. Skipped (assume it
    fits) when `available_memory_bytes` can't answer, as `select_load_dtype`.

    `bf16_bytes` is `_load`'s `exact_bf16_bytes(tensor_infos)` from the GGUF's real shapes (the old
    on-disk-size approximation ran high for a Q4_K_M file and refused a load known to work), so this
    breaker and `select_load_dtype` agree on one number.

    `safety_margin` is the caller's (possibly admin-configured,
    `Settings.memory_safety_margin`/`MATRICXON_MEMORY_SAFETY_MARGIN`) value. `loaded_tags` only
    names what's resident in the raised error. Known gap with on-the-fly dequant
    (`ModelArchitecture._ensure_materialized`): this checks at load time but memory isn't used until
    first materialization, so it is a best-effort guard against "too big for this machine", not a
    guarantee against every interleaving of concurrent loads.
    """
    # Resolved at call time (not a default argument) so tests can patch the module's name.
    # `available_fn`/`memory_kind` let a GPU load check free VRAM instead of system RAM.
    available = (available_fn or available_memory_bytes)()
    if available is None:
        return
    minimum_needed = bf16_bytes * safety_margin
    if minimum_needed <= available:
        return
    loaded = ", ".join(sorted(loaded_tags)) or "none"
    raise InsufficientMemoryError(
        f"not enough {memory_kind} to load {tag!r}: need ~{minimum_needed / 1e9:.1f}GB "
        f"(bf16 estimate, {safety_margin}x safety margin), only "
        f"{available / 1e9:.1f}GB available. Currently loaded: {loaded} - "
        "unload one first, or lower max_loaded_models."
    )

from app.models.load_dtype import available_memory_bytes
from app.server.errors import InsufficientMemoryError


def ensure_enough_memory_to_load(
    tag: str, bf16_bytes: int, safety_margin: float, loaded_tags: list[str]
) -> None:
    """A hard circuit breaker in front of a model load, not just a dtype choice: `select_load_dtype`
    already picks the smallest available footprint (bf16) when float32 wouldn't fit, but it never
    refuses to load outright - so on a machine that's genuinely too tight (e.g. `max_loaded_models`
    raised past the pAIring-matching default of 2 on a RAM-constrained box), it would previously
    attempt the load anyway and risk the OS OOM-killing the whole process rather than this one
    request failing cleanly. Called from `ModelManager._load` after eviction already happened in
    `get_or_load`, so `available_memory_bytes()` here reflects whatever that eviction just freed.
    Skipped (assume it fits) when `available_memory_bytes` can't answer - same policy
    `select_load_dtype` already uses for the same reason.

    `bf16_bytes` is `_load`'s own `exact_bf16_bytes(tensor_infos)` result, computed from the GGUF
    file's real per-tensor shapes rather than approximated from its on-disk size (see
    `exact_bf16_bytes`'s own docstring for why the old on-disk-based approximation ran higher than
    reality for a real Q4_K_M file, refusing a load that had already been confirmed to work) - this
    circuit breaker and `select_load_dtype` now always agree on the same number.

    `safety_margin` is the caller's own (possibly admin-configured, see
    `Settings.memory_safety_margin`/`MATRICXON_MEMORY_SAFETY_MARGIN`) value, not a module constant,
    so a machine that's consistently tighter or looser than the 1.5x default doesn't need a code
    change to reflect that. `loaded_tags` names what's currently resident in the raised error only
    - a known, accepted gap since on-the-fly dequant (see `ModelArchitecture._ensure_materialized`):
    this checks memory at *load* time, but real memory isn't actually consumed until first
    materialization, which can happen meaningfully later - a heuristic best-effort guard against the
    common "this model is just too big for this machine" case, not a hard guarantee against every
    possible interleaving of concurrent loads.
    """
    available = available_memory_bytes()
    if available is None:
        return
    minimum_needed = bf16_bytes * safety_margin
    if minimum_needed <= available:
        return
    loaded = ", ".join(sorted(loaded_tags)) or "none"
    raise InsufficientMemoryError(
        f"not enough memory to load {tag!r}: need ~{minimum_needed / 1e9:.1f}GB "
        f"(bf16 estimate, {safety_margin}x safety margin), only "
        f"{available / 1e9:.1f}GB available. Currently loaded: {loaded} - "
        "unload one first, or lower max_loaded_models."
    )

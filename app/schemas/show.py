from pydantic import BaseModel

from app.schemas.tags import ModelDetails


class ShowRequest(BaseModel):
    model: str


class ShowResponse(BaseModel):
    capabilities: list[str]
    details: ModelDetails
    # Matricxon's own real, current minimum RAM to actually load this exact tag - the same real
    # number `ModelManager._ensure_enough_memory_to_load` checks a real request against (exact
    # per-tensor bf16 footprint x the configured safety margin - see
    # `app.models.load_dtype.exact_bf16_bytes`/`Settings.memory_safety_margin`), not a generic
    # on-disk-size-based estimate. Real, confirmed gap this closes (2026-09-21): matricxon always
    # dequantizes to bf16/float32 before computing (see ROADMAP.md's "Known issues"), so its real
    # RAM need for a real Q4_K_M file runs meaningfully higher than an engine (like Ollama) that
    # computes directly on the quantized blocks and never expands them - a caller showing one
    # generic "min RAM" figure for a file regardless of which engine will actually run it
    # (confirmed live: pAIring's own static catalog `min_ram_gb` is accurate for Ollama, not for
    # Matricxon, on the exact same real Ministral-3B file) needs this to show the right one.
    estimated_ram_gb: float

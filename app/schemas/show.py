from pydantic import BaseModel

from app.schemas.tags import ModelDetails


class ShowRequest(BaseModel):
    model: str


class ShowResponse(BaseModel):
    capabilities: list[str]
    details: ModelDetails
    # Matricxon's minimum RAM to load this exact tag: the number
    # `ModelManager._ensure_enough_memory_to_load` checks a request against (exact per-tensor bf16
    # footprint x the configured safety margin;
    # `app.models.load_dtype.exact_bf16_bytes`/`Settings.memory_safety_margin`), not an on-disk-size
    # estimate. Matricxon dequantizes to bf16/float32 before computing, so its need for a Q4_K_M
    # file is well above an engine (like Ollama) computing on quantized blocks; a caller showing one
    # generic "min RAM" for every engine (pAIring's static catalog `min_ram_gb` suits Ollama, not
    # Matricxon, on the same Ministral-3B file) needs this.
    estimated_ram_gb: float

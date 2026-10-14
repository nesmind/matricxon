"""GET /api/node/load: how busy this node is, for cluster routing (pAIring's pool)."""

from pydantic import BaseModel


class LoadedModelLoad(BaseModel):
    name: str
    size_bytes: int
    active: int  # replies queued or running on this model


class NodeLoadResponse(BaseModel):
    active: int  # replies queued or running on every model, from every client
    models: list[LoadedModelLoad]
    threads: int  # compute threads (MATRICXON_TORCH_THREADS, else CPU count)
    free_memory_bytes: int | None  # MemAvailable; None when unknown
    node_kind: str  # "compute" or "voter"

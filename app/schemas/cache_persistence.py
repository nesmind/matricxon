from pydantic import BaseModel, Field


class CachePersistenceStatus(BaseModel):
    enabled: bool
    budget_mb: int
    ttl_hours: int
    files: int
    used_bytes: int


class CachePersistenceUpdate(BaseModel):
    """Fields left out stay unchanged. Turning `enabled` off also deletes every stored cache."""

    enabled: bool | None = None
    budget_mb: int | None = Field(default=None, ge=0)
    ttl_hours: int | None = Field(default=None, ge=1)

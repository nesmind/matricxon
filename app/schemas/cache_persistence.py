from pydantic import BaseModel, Field


class CachePersistenceStatus(BaseModel):
    enabled: bool
    budget_mb: int
    ttl_hours: int
    files: int
    # Distinct chats with a saved copy (`files` can be more: branches of one chat).
    chats: int = 0
    used_bytes: int
    folder: str | None = None
    shared: bool = False


class CachePersistenceUpdate(BaseModel):
    """Omitted fields stay unchanged. Turning `enabled` off also deletes every stored cache."""

    enabled: bool | None = None
    budget_mb: int | None = Field(default=None, ge=0)
    ttl_hours: int | None = Field(default=None, ge=1)

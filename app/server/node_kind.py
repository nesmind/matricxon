from fastapi import HTTPException

from app.config import settings


def require_compute() -> None:
    """FastAPI dependency for routes that load or serve models: a voter-only node has none."""
    if settings.node_kind == "voter":
        raise HTTPException(
            status_code=409,
            detail="this node is voter-only (consensus only): it does not load or serve models",
        )

from fastapi import APIRouter, Depends

from app.dependencies import get_cache_store
from app.runtime.cache_store import PersistentCacheStore
from app.schemas.cache_persistence import CachePersistenceStatus, CachePersistenceUpdate

router = APIRouter()


class CachePersistenceHandler:
    """Admin view of the encrypted on-disk prompt cache: its limits, what it holds, and a way to
    change or empty it. Like /api/pull and /api/delete this has no auth of its own - pAIring gates
    it behind its admin-only Settings page."""

    def __init__(self, store: PersistentCacheStore) -> None:
        self._store = store

    def status(self) -> CachePersistenceStatus:
        files, used = self._store.usage()
        policy = self._store.policy
        return CachePersistenceStatus(
            enabled=policy.enabled,
            budget_mb=policy.budget_mb,
            ttl_hours=policy.ttl_hours,
            files=files,
            used_bytes=used,
        )

    def update(self, change: CachePersistenceUpdate) -> CachePersistenceStatus:
        self._store.policy.update(change.enabled, change.budget_mb, change.ttl_hours)
        if not self._store.policy.enabled:
            self._store.clear()
        self._store.enforce_limits()
        return self.status()

    def forget_chat(self, tag: str) -> CachePersistenceStatus:
        self._store.forget(tag)
        return self.status()

    def clear(self) -> CachePersistenceStatus:
        self._store.clear()
        return self.status()


@router.get("/api/cache/persistence", response_model=CachePersistenceStatus)
def get_cache_persistence(
    store: PersistentCacheStore = Depends(get_cache_store),
) -> CachePersistenceStatus:
    return CachePersistenceHandler(store).status()


@router.put("/api/cache/persistence", response_model=CachePersistenceStatus)
def put_cache_persistence(
    change: CachePersistenceUpdate, store: PersistentCacheStore = Depends(get_cache_store)
) -> CachePersistenceStatus:
    return CachePersistenceHandler(store).update(change)


@router.delete("/api/cache/persistence", response_model=CachePersistenceStatus)
def delete_cache_persistence(
    store: PersistentCacheStore = Depends(get_cache_store),
) -> CachePersistenceStatus:
    return CachePersistenceHandler(store).clear()


@router.delete("/api/cache/persistence/chats/{tag}", response_model=CachePersistenceStatus)
def delete_chat_cache(
    tag: str, store: PersistentCacheStore = Depends(get_cache_store)
) -> CachePersistenceStatus:
    """Drops the stored prompt caches of one conversation (its `cache_tag`), e.g. after pAIring
    deletes the chat."""
    return CachePersistenceHandler(store).forget_chat(tag)

import logging

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.cache_miss_diagnosis import CacheMissDiagnosis
from app.runtime.cache_store import BLOCK_TOKENS, PersistentCacheStore
from app.runtime.prompt_slot import CacheSlot
from app.runtime.slot_codec import SlotCodec

logger = logging.getLogger(__name__)


class ModelCacheTier:
    """One loaded model's view of the on-disk store: its `PromptCache` spills evicted slots here
    and asks here for a conversation it no longer holds in RAM. `model_id` ties files to exact
    weights (tag + file size), so a replaced model never picks up another's cache."""

    def __init__(self, store: PersistentCacheStore, model_id: str) -> None:
        self._store = store
        self._model_id = model_id

    def spill(self, slot: CacheSlot) -> None:
        self._store.spill_async(self._model_id, slot)

    def restore(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        num_ctx: int,
        dtype: torch.dtype,
        min_tokens: int,
    ) -> CacheSlot | None:
        """A slot rebuilt from disk that shares more than `min_tokens` leading tokens with
        `prompt_ids`, or None (nothing stored, or the file no longer fits this model)."""
        payload = self._store.take(self._model_id, prompt_ids, dtype, min_tokens)
        if payload is None:
            return None
        try:
            slot = SlotCodec.decode(payload, architecture, num_ctx, dtype)
        except (ValueError, KeyError, RuntimeError):
            logger.warning("a stored prompt cache no longer fits its model, ignoring it")
            return None
        logger.info("restored a %d-token prompt cache from disk", len(slot.token_ids))
        return slot

    def explain(self, prompt_ids: list[int], dtype: torch.dtype, tag: str) -> str:
        """Why the disk tier had nothing for this prompt (for the log)."""
        entries, wanted = self._store.inspect(prompt_ids)
        return CacheMissDiagnosis.disk(
            entries,
            wanted,
            BLOCK_TOKENS,
            self._model_id,
            dtype,
            tag,
            self._store.policy.enabled,
        )

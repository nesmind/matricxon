import functools
import logging
import threading
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.cache_miss_diagnosis import CacheMissDiagnosis
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.kv_cache import KVCache
from app.runtime.mamba_cache import NemotronHHybridCache
from app.runtime.prompt_slot import CacheSlot

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _locked(
    method: Callable[Concatenate["PromptCache", P], R],
) -> Callable[Concatenate["PromptCache", P], R]:
    """Runs a PromptCache method under its lock: the worker thread uses the pool while HTTP threads
    ask about, or spill, a chat (cluster routing)."""

    @functools.wraps(method)
    def wrapper(self: "PromptCache", *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class PromptCache:
    """A model's cached conversations, so a chat's next turn computes only new tokens. pAIring
    resends the whole conversation each turn; without this every turn re-ran prefill over all of it
    (turn 2 of a two-turn chat prefilled 70 tokens in 12 s, where Ollama, reusing its cache,
    answered in 3.6 s).

    Several users share one loaded model, so this keeps a pool of slots (one per recent
    conversation) instead of one cache every request would overwrite. A new prompt is matched
    against every slot on exact token ids (a reply that re-tokenizes differently just ends the match
    early) and takes the slot sharing the longest prefix. If using it would discard cached tokens
    another conversation might need (it shares only a prefix, or the tail differs) and there is
    room, the shared prefix is copied into a new slot (`CacheSlot.fork`: a memory copy, far cheaper
    than recomputing); with no room it is reused in place.

    A plain `KVCache` can be cut back to any prefix length. A hybrid cache (`NemotronHHybridCache`:
    Mamba-2 / Gated DeltaNet recurrent state) can't, since that state exists only at the current
    position: during prefill `ChatEngine` calls `capture` at a few positions, each slot keeps its
    newest `MAX_SNAPSHOTS`, and the next call restores the latest snapshot at or before the shared
    prefix and recomputes from there.

    The pool is bounded by `max_slots` and `budget_bytes` (caches plus snapshots); the least
    recently used slot goes first, never the one in use. A model whose single cache exceeds the
    budget still keeps that slot, so single-user behavior is never worse than one cache.

    Not reused (a fresh unpooled cache is built, the pool untouched) when the prompt carries image
    embeddings: their placeholder ids are identical for different images, so an id match wouldn't
    mean the same content. Only a different dtype never matches: `num_ctx` is capacity, so a slot
    built for a smaller one is copied into a bigger cache and one built for a larger one is used as
    is.

    One instance per loaded model (`ModelHandle.prompt_cache`), used only from that model's
    `ModelWorker` thread.
    """

    # Newest recurrent snapshots kept per slot (~50MB each for Qwen3.5-4B/9B).
    MAX_SNAPSHOTS = 4
    # Cut-back smaller than this is done in place: forking a copy isn't worth it.
    FORK_MIN_DROPPED_TOKENS = 16
    # A new prompt continues a slot's conversation if it repeats the slot's whole previous prompt
    # except at most this many trailing tokens (the chat template re-renders the end of a turn
    # differently once the reply is in history, e.g. Qwen's empty thinking block).
    CONTINUATION_TAIL_TOKENS = 16

    def __init__(
        self,
        max_slots: int = 4,
        budget_bytes: int = 2 << 30,
        tier: ModelCacheTier | None = None,
        context_shift: bool = True,
    ) -> None:
        self._max_slots = max(1, max_slots)
        self._budget_bytes = budget_bytes
        self._tier = tier
        self._context_shift = context_shift
        self._slots: list[CacheSlot] = []
        self._lock = threading.RLock()
        self._tick = 0
        self.reused_tokens = 0
        self.miss_reason = ""

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @_locked
    def acquire(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        num_ctx: int,
        dtype: torch.dtype,
        reusable: bool,
        tag: str = "",
    ) -> tuple[object, int]:
        """(cache, n): a cache already holding `prompt_ids[:n]`, so prefill starts at n."""
        key = (num_ctx, dtype)
        self.reused_tokens = 0
        self.miss_reason = "" if reusable else "the prompt contains images (never reused)"
        if reusable:
            if self._context_shift and tag:
                self._shift_trimmed_chat(architecture, prompt_ids, key, tag)
            best, common, n = self._best_slot(prompt_ids, key)
            if self._tier is not None:
                restored = self._tier.restore(architecture, prompt_ids, num_ctx, dtype, n)
                if restored is not None:
                    self._add(restored)
                    if self._context_shift and tag:
                        self._shift_trimmed_chat(architecture, prompt_ids, key, tag)
                    best, common, n = self._best_slot(prompt_ids, key)
            if best is not None:
                self.reused_tokens = n
                slot = self._take(best, common, n, num_ctx)
                slot.prompt_len = len(prompt_ids)
                slot.in_use = True
                slot.tag = tag or slot.tag
                return slot.cache, n
            self.miss_reason = self._explain_miss(prompt_ids, key, tag)
        cache = architecture.build_cache(max_seq_len=num_ctx, dtype=dtype)
        if reusable and isinstance(cache, (KVCache, NemotronHHybridCache)):
            slot = self._add(CacheSlot(cache, key, prompt_len=len(prompt_ids), tag=tag))
            slot.in_use = True
        return cache, 0

    def _shift_trimmed_chat(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        key: tuple[int, torch.dtype],
        tag: str,
    ) -> None:
        """If this chat's cached prompt is the new one with its oldest messages trimmed away, cuts
        those tokens out of the cache (re-rotating the rest) instead of re-reading the chat.
        Approximate: the kept tokens were read while the dropped ones were visible, so they carry a
        trace of them, the trade llama.cpp's context shifting makes."""
        for slot in self._slots:
            if slot.tag != tag or slot.key[1] != key[1] or slot.in_use:
                continue
            gap = slot.find_gap(prompt_ids)
            if gap is None:
                continue
            keep, dropped, run = gap
            rotations = architecture.context_shift_rotations(dropped)
            if rotations is None:
                return
            slot.shift_out(keep, dropped, rotations)
            logger.info(
                "context shift: dropped %d trimmed tokens after the first %d, kept the next %d",
                dropped,
                keep,
                run,
            )
            return

    def _explain_miss(self, prompt_ids: list[int], key: tuple[int, torch.dtype], tag: str) -> str:
        parts = [CacheMissDiagnosis.ram(self._slots, prompt_ids, key, tag)]
        if self._tier is not None:
            parts.append(self._tier.explain(prompt_ids, key[1], tag))
        return "; ".join(
            f"{label}: {text}" for label, text in zip(("RAM", "disk"), parts, strict=False)
        )

    def _best_slot(
        self, prompt_ids: list[int], key: tuple[int, torch.dtype]
    ) -> tuple[CacheSlot | None, int, int]:
        """The slot supplying the most tokens: (slot, common, reusable), or (None, 0, 0)."""
        best, best_common, best_n = None, 0, 0
        for slot in self._slots:
            if slot.key[1] != key[1] or (slot.in_use and self._max_slots == 1):
                continue
            common, n = slot.match(prompt_ids)
            newer = best is not None and slot.last_used > best.last_used
            if n > 0 and (best is None or n > best_n or (n == best_n and newer)):
                best, best_common, best_n = slot, common, n
        return best, best_common, best_n

    def _take(self, slot: CacheSlot, common: int, n: int, num_ctx: int) -> CacheSlot:
        """Makes `slot`'s first `n` tokens the cache to prefill into and returns the slot holding
        it. In place when the prompt continues this slot's conversation (only the old reply and the
        end of the last turn are dropped) or little is dropped; otherwise, and always when another
        reply runs on the slot, a fork, so the slot keeps serving the conversation this prompt
        doesn't continue. Forks are trimmed to the limits by evicting the least recently used idle
        slot."""
        tail = min(self.CONTINUATION_TAIL_TOKENS, slot.prompt_len // 4)
        continues = common >= slot.prompt_len - tail
        dropped = len(slot.token_ids) - n
        forkable = self._max_slots > 1
        wanted = slot.in_use or (not continues and dropped >= self.FORK_MIN_DROPPED_TOKENS)
        if slot.key[0] < num_ctx:
            bigger = slot.fork(n, num_ctx)
            if continues and not slot.in_use:
                self._slots.remove(slot)
            return self._add(bigger)
        if forkable and wanted:
            return self._add(slot.fork(n))
        slot.rewind(n)
        self._use(slot)
        return slot

    def _use(self, slot: CacheSlot) -> None:
        self._tick += 1
        slot.last_used = self._tick

    def _add(self, slot: CacheSlot) -> CacheSlot:
        self._slots.append(slot)
        self._use(slot)
        self._evict_over_limits(keep=slot)
        return slot

    def _evict_over_limits(self, keep: CacheSlot | None = None) -> None:
        """Drops least-recently-used idle slots (never one in use, nor `keep`) until within both
        limits or none are left to drop."""
        while True:
            over_count = len(self._slots) > self._max_slots
            over_budget = sum(s.nbytes() for s in self._slots) > self._budget_bytes
            idle = [s for s in self._slots if not s.in_use and s is not keep]
            if not (over_count or over_budget) or not idle or len(self._slots) <= 1:
                return
            victim = min(idle, key=lambda s: s.last_used)
            self._slots.remove(victim)
            if self._tier is not None:
                self._tier.spill(victim)

    def _slot_of(self, cache: object) -> CacheSlot | None:
        return next((s for s in self._slots if s.cache is cache), None)

    @_locked
    def advanced(self, cache: object, token_ids: list[int]) -> None:
        """Records tokens whose keys/values were just committed to `cache` (after
        `KVCache.advance`). A no-op for a cache the pool isn't holding."""
        slot = self._slot_of(cache)
        if slot is not None:
            slot.token_ids.extend(token_ids)

    @_locked
    def capture(self, cache: object) -> None:
        """Snapshots a hybrid `cache`'s recurrent state at its current length (a no-op for any other
        cache, or one the pool isn't holding), keeping the slot's newest few."""
        slot = self._slot_of(cache)
        if slot is None or not slot.hybrid:
            return
        slot.snapshots[cache.length] = cache.snapshot()
        for pos in sorted(slot.snapshots)[: -self.MAX_SNAPSHOTS]:
            del slot.snapshots[pos]
        self._evict_over_limits(keep=slot)

    @_locked
    def release(self, cache: object) -> None:
        """The reply that acquired `cache` has finished (or stopped): its slot may be evicted or
        reused again, and any limit exceeded while it was busy is enforced now."""
        slot = self._slot_of(cache)
        if slot is not None:
            slot.in_use = False
            self._evict_over_limits()

    @_locked
    def holds(self, tag: str) -> bool:
        """Does this pool hold a conversation of the chat `tag` (idle or in use)?"""
        return bool(tag) and any(slot.tag == tag for slot in self._slots)

    @_locked
    def spill_chat(self, tag: str) -> int:
        """Moves the chat's idle conversations from RAM to the disk tier (the model stays loaded) so
        another node can pick them up. Returns how many; one still replying stays."""
        if self._tier is None or not tag:
            return 0
        moving = [s for s in self._slots if s.tag == tag and not s.in_use]
        for slot in moving:
            self._slots.remove(slot)
            self._tier.spill(slot)
        return len(moving)

    @_locked
    def persist_all(self) -> None:
        """Queues every idle slot for the disk tier: the model is unloading or the server shutting
        down, so nothing else will use these RAM copies."""
        if self._tier is not None:
            for slot in self._slots:
                if not slot.in_use:
                    self._tier.spill(slot)

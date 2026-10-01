from dataclasses import dataclass, field

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.kv_cache import KVCache
from app.runtime.mamba_cache import HybridSnapshot, NemotronHHybridCache


@dataclass
class _Slot:
    """One cached conversation: its cache, the exact token ids that cache holds (positions
    0..len-1), and - for a hybrid cache - the recurrent-state snapshots it can be rewound to."""

    cache: KVCache | NemotronHHybridCache
    key: tuple[int, torch.dtype]
    token_ids: list[int] = field(default_factory=list)
    snapshots: dict[int, HybridSnapshot] = field(default_factory=dict)
    last_used: int = 0
    prompt_len: int = 0  # length of the prompt this slot was last prefilled for
    in_use: bool = False  # a reply is running on this slot's cache right now

    @property
    def hybrid(self) -> bool:
        return isinstance(self.cache, NemotronHHybridCache)

    def match(self, prompt_ids: list[int]) -> tuple[int, int]:
        """(common, reusable): how many leading prompt tokens equal this slot's tokens, and how
        many of them it can supply without recomputing. At least the prompt's last token is always
        recomputed (its logits start the reply). A plain KV cache can be cut back to any length; a
        hybrid one only to a recurrent-state snapshot."""
        common = 0
        for cached, new in zip(self.token_ids, prompt_ids, strict=False):
            if cached != new:
                break
            common += 1
        common = min(common, len(prompt_ids) - 1)
        if not self.hybrid:
            return common, common
        return common, max((pos for pos in self.snapshots if pos <= common), default=0)

    def nbytes(self) -> int:
        return self.cache.nbytes() + sum(s.nbytes() for s in self.snapshots.values())

    def rewind(self, n: int) -> None:
        """Cuts this slot back to its first `n` tokens, in place."""
        if self.hybrid:
            self.cache.restore(self.snapshots[n])
            self.snapshots = {p: s for p, s in self.snapshots.items() if p <= n}
        else:
            self.cache.truncate(n)
        del self.token_ids[n:]

    def fork(self, n: int) -> "_Slot":
        """A new slot holding a copy of the first `n` tokens; this slot is left untouched."""
        if self.hybrid:
            cache = self.cache.fork_from(self.snapshots[n])
        else:
            cache = self.cache.fork(n)
        snapshots = {p: s for p, s in self.snapshots.items() if p <= n}
        return _Slot(cache, self.key, self.token_ids[:n], snapshots, prompt_len=self.prompt_len)


class PromptCache:
    """A model's cached conversations, so a chat's next turn only computes the tokens that are
    new. pAIring resends the whole conversation every turn; before this every turn re-ran prefill
    over all of it (measured 2026-09-23: turn 2 of a two-turn chat prefilled 70 tokens in 12 s,
    while Ollama - which reuses its cache the same way - answered in 3.6 s).

    Several users share one loaded model, so this keeps a small pool of slots (one per recent
    conversation), not a single cache that every other user's request would overwrite. A new
    prompt is matched against every slot on its exact token ids - a token-level match, so a reply
    that re-tokenizes differently just ends the match early instead of reusing anything wrong -
    and takes the slot sharing the longest prefix. If using it would throw away cached tokens
    another conversation might still need (it only shares a prefix, or the tail differs) and there
    is room, the shared prefix is copied into a new slot instead (`_Slot.fork` - a memory copy,
    far cheaper than recomputing); with no room it is reused in place.

    A plain `KVCache` can be cut back to any prefix length. A hybrid cache (`NemotronHHybridCache`:
    Mamba-2 / Gated DeltaNet recurrent state) can't - recurrent state only exists at the current
    position - so while prefilling, `ChatEngine` calls `capture()` at a few positions and each
    slot keeps its newest `MAX_SNAPSHOTS`; the next call restores the latest snapshot at or before
    the shared prefix and recomputes from there.

    The pool is bounded by `max_slots` and by `budget_bytes` (caches plus snapshots); the least
    recently used slot goes first, never the one in use. A model whose single cache already
    exceeds the budget still keeps that one slot, so single-user behavior is never worse than one
    cache.

    Not reused (a fresh, unpooled cache is built, and the pool is left alone) when the prompt
    carries image embeddings - their placeholder token ids are identical for different images, so
    an id match wouldn't mean the same content. A different `num_ctx` or dtype never matches a
    slot.

    One instance per loaded model (`ModelHandle.prompt_cache`); only ever used from that model's
    own `ModelWorker` thread, which runs one generation at a time.
    """

    #: Newest recurrent snapshots kept per slot (~50MB each for Qwen3.5-4B/9B).
    MAX_SNAPSHOTS = 4
    #: Cut-back smaller than this is done in place - forking a copy wouldn't be worth it.
    FORK_MIN_DROPPED_TOKENS = 16
    #: A new prompt continues a slot's conversation if it repeats the slot's whole previous prompt
    #: except at most this many trailing tokens (the chat template re-renders the end of a turn
    #: differently once the reply is in the history, e.g. Qwen's empty thinking block).
    CONTINUATION_TAIL_TOKENS = 16

    def __init__(self, max_slots: int = 4, budget_bytes: int = 2 << 30) -> None:
        self._max_slots = max(1, max_slots)
        self._budget_bytes = budget_bytes
        self._slots: list[_Slot] = []
        self._tick = 0
        self.reused_tokens = 0

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    def acquire(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        num_ctx: int,
        dtype: torch.dtype,
        reusable: bool,
    ) -> tuple[object, int]:
        """(cache, n): a cache already holding `prompt_ids[:n]`, so prefill starts at n."""
        key = (num_ctx, dtype)
        self.reused_tokens = 0
        if reusable:
            best, common, n = self._best_slot(prompt_ids, key)
            if best is not None:
                self.reused_tokens = n
                slot = self._take(best, common, n)
                slot.prompt_len = len(prompt_ids)
                slot.in_use = True
                return slot.cache, n
        cache = architecture.build_cache(max_seq_len=num_ctx, dtype=dtype)
        if reusable and isinstance(cache, (KVCache, NemotronHHybridCache)):
            slot = self._add(_Slot(cache, key, prompt_len=len(prompt_ids)))
            slot.in_use = True
        return cache, 0

    def _best_slot(
        self, prompt_ids: list[int], key: tuple[int, torch.dtype]
    ) -> tuple[_Slot | None, int, int]:
        """The slot supplying the most tokens: (slot, common, reusable), or (None, 0, 0)."""
        best, best_common, best_n = None, 0, 0
        for slot in self._slots:
            if slot.key != key or (slot.in_use and self._max_slots == 1):
                continue
            common, n = slot.match(prompt_ids)
            newer = best is not None and slot.last_used > best.last_used
            if n > 0 and (best is None or n > best_n or (n == best_n and newer)):
                best, best_common, best_n = slot, common, n
        return best, best_common, best_n

    def _take(self, slot: _Slot, common: int, n: int) -> _Slot:
        """Makes `slot`'s first `n` tokens the cache to prefill into and returns the slot that
        holds it. In place when the prompt continues this slot's conversation (only the old reply
        and the end of the last turn are dropped) or when little is dropped; otherwise - and
        always when another reply is running on the slot - a fork, so the slot keeps serving the
        conversation this prompt does not continue. Forks are trimmed back to the limits by
        evicting the least recently used idle slot."""
        tail = min(self.CONTINUATION_TAIL_TOKENS, slot.prompt_len // 4)
        continues = common >= slot.prompt_len - tail
        dropped = len(slot.token_ids) - n
        forkable = self._max_slots > 1
        wanted = slot.in_use or (not continues and dropped >= self.FORK_MIN_DROPPED_TOKENS)
        if forkable and wanted:
            return self._add(slot.fork(n))
        slot.rewind(n)
        self._use(slot)
        return slot

    def _use(self, slot: _Slot) -> None:
        self._tick += 1
        slot.last_used = self._tick

    def _add(self, slot: _Slot) -> _Slot:
        self._slots.append(slot)
        self._use(slot)
        self._evict_over_limits(keep=slot)
        return slot

    def _evict_over_limits(self, keep: _Slot | None = None) -> None:
        """Drops least-recently-used idle slots (never one in use, nor `keep`) until within both
        limits - or until none are left to drop."""
        while True:
            over_count = len(self._slots) > self._max_slots
            over_budget = sum(s.nbytes() for s in self._slots) > self._budget_bytes
            idle = [s for s in self._slots if not s.in_use and s is not keep]
            if not (over_count or over_budget) or not idle or len(self._slots) <= 1:
                return
            self._slots.remove(min(idle, key=lambda s: s.last_used))

    def _slot_of(self, cache: object) -> _Slot | None:
        return next((s for s in self._slots if s.cache is cache), None)

    def advanced(self, cache: object, token_ids: list[int]) -> None:
        """Records tokens whose keys/values were just committed to `cache` (after
        `KVCache.advance`). A no-op for a cache the pool isn't holding."""
        slot = self._slot_of(cache)
        if slot is not None:
            slot.token_ids.extend(token_ids)

    def capture(self, cache: object) -> None:
        """Snapshots a hybrid `cache`'s recurrent state at its current length (a no-op for any
        other cache, or one the pool isn't holding), keeping the slot's newest few."""
        slot = self._slot_of(cache)
        if slot is None or not slot.hybrid:
            return
        slot.snapshots[cache.length] = cache.snapshot()
        for pos in sorted(slot.snapshots)[: -self.MAX_SNAPSHOTS]:
            del slot.snapshots[pos]
        self._evict_over_limits(keep=slot)

    def release(self, cache: object) -> None:
        """The reply that acquired `cache` has finished (or stopped): its slot may be evicted or
        reused again, and any limit that was exceeded while it was busy is enforced now."""
        slot = self._slot_of(cache)
        if slot is not None:
            slot.in_use = False
            self._evict_over_limits()

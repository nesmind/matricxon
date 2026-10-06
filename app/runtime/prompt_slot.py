from dataclasses import dataclass, field

import torch

from app.runtime.kv_cache import KVCache
from app.runtime.mamba_cache import HybridSnapshot, NemotronHHybridCache


@dataclass
class CacheSlot:
    """One cached conversation: its cache, the exact token ids that cache holds (positions
    0..len-1), and - for a hybrid cache - the recurrent-state snapshots it can be rewound to."""

    cache: KVCache | NemotronHHybridCache
    key: tuple[int, torch.dtype]
    token_ids: list[int] = field(default_factory=list)
    snapshots: dict[int, HybridSnapshot] = field(default_factory=dict)
    last_used: int = 0
    prompt_len: int = 0  # length of the prompt this slot was last prefilled for
    in_use: bool = False  # a reply is running on this slot's cache right now
    tag: str = ""  # the conversation it belongs to (opaque; lets a deleted chat's cache be dropped)

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

    # : A gap must be followed by at least this many tokens that line up again, else it is not
    # trusted.
    MIN_RUN_AFTER_GAP = 32

    def find_gap(self, prompt_ids: list[int]) -> tuple[int, int, int] | None:
        """`(keep, gap, run)` when `prompt_ids` is this slot's tokens with `gap` tokens removed
        right
        after the first `keep` (the chat's oldest messages were trimmed away) and the next `run`
        tokens
        then line up again; else None. Plain caches only - a recurrent state can't be cut."""
        old = self.token_ids
        if self.hybrid or not old:
            return None
        keep = 0
        for cached, new in zip(old, prompt_ids, strict=False):
            if cached != new:
                break
            keep += 1
        if keep == 0 or keep >= len(old) or keep >= len(prompt_ids):
            return None
        best: tuple[int, int, int] | None = None
        start = keep
        while True:
            try:
                start = old.index(prompt_ids[keep], start + 1)
            except ValueError:
                return best
            run = 0
            for cached, new in zip(old[start:], prompt_ids[keep:], strict=False):
                if cached != new:
                    break
                run += 1
            if run >= self.MIN_RUN_AFTER_GAP and (best is None or run > best[2]):
                best = (keep, start - keep, run)

    def shift_out(self, keep: int, gap: int, rotations: list) -> None:
        """Drops `gap` tokens after the first `keep`, in place (see `KVCache.drop_range`)."""
        self.cache.drop_range(keep, gap, rotations)
        del self.token_ids[keep : keep + gap]
        self.prompt_len = max(self.prompt_len - gap, 0)

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

    def fork(self, n: int, num_ctx: int | None = None) -> "CacheSlot":
        """A new slot holding a copy of the first `n` tokens; this slot is left untouched. `num_ctx`
        gives the copy a different capacity."""
        if self.hybrid:
            cache = self.cache.fork_from(self.snapshots[n], num_ctx)
        else:
            cache = self.cache.fork(n, num_ctx)
        snapshots = {p: s for p, s in self.snapshots.items() if p <= n}
        key = (num_ctx or self.key[0], self.key[1])
        return CacheSlot(
            cache, key, self.token_ids[:n], snapshots, prompt_len=self.prompt_len, tag=self.tag
        )

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

    def fork(self, n: int) -> "CacheSlot":
        """A new slot holding a copy of the first `n` tokens; this slot is left untouched."""
        if self.hybrid:
            cache = self.cache.fork_from(self.snapshots[n])
        else:
            cache = self.cache.fork(n)
        snapshots = {p: s for p, s in self.snapshots.items() if p <= n}
        return CacheSlot(cache, self.key, self.token_ids[:n], snapshots, prompt_len=self.prompt_len)

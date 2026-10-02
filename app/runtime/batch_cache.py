"""Wrappers that let ONE forward pass run a decode step for several sequences at once.

A batched decode step feeds `B` sequences' newest tokens through the model as a `(B, 1)` input. The
linear layers see `B` tokens and read each weight once for all of them (decode is memory-bound, so
that is where the throughput comes from); attention and recurrent state stay per sequence. These
wrappers stand in for the usual cache object inside the layers' unchanged code:

- `length` is a `(B, 1, 1, 1)` tensor of each sequence's current length, so an attention module's
  own `kv_idx <= q_idx + cache_offset` mask comes out per sequence, shape `(B, 1, 1, kv_len)`;
- `update` appends each sequence's new key/value to ITS cache and returns the caches padded to a
  common length, which that mask hides.

After the forward the caller advances each real cache by one, as for an ordinary step.
"""

import torch

from app.runtime.kv_cache import KVCache
from app.runtime.mamba_cache import NemotronHHybridCache


def _padded(parts: list[torch.Tensor]) -> torch.Tensor:
    """Stacks `(1, heads, L_b, dim)` tensors into `(B, heads, max L, dim)`, zero-padded."""
    longest = max(p.shape[2] for p in parts)
    out = parts[0].new_zeros((len(parts), parts[0].shape[1], longest, parts[0].shape[3]))
    for b, part in enumerate(parts):
        out[b, :, : part.shape[2]] = part[0]
    return out


class BatchedKVCache:
    batched = True

    def __init__(self, caches: list[KVCache]) -> None:
        self._caches = caches
        self._device = caches[0].device

    @property
    def length(self) -> torch.Tensor:
        return torch.tensor([c.length for c in self._caches], device=self._device).view(-1, 1, 1, 1)

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = [
            cache.update(layer_idx, k[b : b + 1], v[b : b + 1])
            for b, cache in enumerate(self._caches)
        ]
        return _padded([p[0] for p in pairs]), _padded([p[1] for p in pairs])


class BatchedHybridCache:
    """The same for `NemotronHHybridCache`s (Mamba-2 / Gated DeltaNet layers plus attention)."""

    batched = True

    def __init__(self, caches: list[NemotronHHybridCache]) -> None:
        self._caches = caches
        self._device = caches[0].device

    @property
    def length(self) -> torch.Tensor:
        return torch.tensor([c.length for c in self._caches], device=self._device).view(-1, 1, 1, 1)

    def update_attention(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = [
            cache.update_attention(layer_idx, k[b : b + 1], v[b : b + 1])
            for b, cache in enumerate(self._caches)
        ]
        return _padded([p[0] for p in pairs]), _padded([p[1] for p in pairs])

    def mamba_state(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        states = [cache.mamba_state(layer_idx) for cache in self._caches]
        return torch.cat([s[0] for s in states]), torch.cat([s[1] for s in states])

    def set_mamba_state(
        self, layer_idx: int, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> None:
        for b, cache in enumerate(self._caches):
            cache.set_mamba_state(layer_idx, conv_state[b : b + 1], ssm_state[b : b + 1])


def batch_cache_for(caches: list[object]) -> BatchedKVCache | BatchedHybridCache | None:
    """The wrapper for `caches`, or None when they aren't all one batchable kind."""
    if all(isinstance(c, KVCache) for c in caches):
        return BatchedKVCache(caches)
    if all(isinstance(c, NemotronHHybridCache) for c in caches):
        return BatchedHybridCache(caches)
    return None

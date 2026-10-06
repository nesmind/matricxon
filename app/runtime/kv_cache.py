import torch

from app.architectures.rope import rotate_half
from app.server.errors import PromptTooLongError


class KVCache:
    """Pre-allocated per-layer key/value cache for autoregressive generation.

    pAIring resends the full history every call; `PromptCache`
    (app/runtime/prompt_cache.py) keeps one of these alive between calls and
    `truncate()`s it back to the prefix the next prompt shares, so only the
    new tokens get computed.

    Writes are keyed by absolute position via `update()`, but the cache's
    committed length only moves forward on `advance()` - every layer in one
    forward pass must see the same starting offset, and only the caller
    (after every layer has run) knows the step is actually done.

    `layer_shapes` is a `(n_head_kv, head_dim)` pair *per layer*, not one
    shared pair for the whole model - most architectures (mistral3, llama)
    just repeat the same pair `n_layer` times, but Gemma4's real per-layer-
    type head dims (8 kv-heads of dim 256 on local/sliding layers, 1 of dim
    512 - effectively MQA - on global layers, confirmed via its real GGUF
    tensor shapes) genuinely need a separate tensor per layer rather than
    one shared `(n_layer, ...)` tensor, which is why this stores a list of
    per-layer tensors instead of the single stacked tensor M4 originally
    built.
    """

    def __init__(
        self,
        layer_shapes: list[tuple[int, int]],
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self._k = [
            torch.zeros((1, n_head_kv, max_seq_len, head_dim), dtype=dtype, device=device)
            for n_head_kv, head_dim in layer_shapes
        ]
        self._v = [
            torch.zeros((1, n_head_kv, max_seq_len, head_dim), dtype=dtype, device=device)
            for n_head_kv, head_dim in layer_shapes
        ]
        self._device = torch.device(device)
        self._layer_shapes = list(layer_shapes)
        self._dtype = dtype
        self._max_seq_len = max_seq_len
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def device(self) -> torch.device:
        return self._device

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes `k`/`v` (batch, n_head_kv, n_new, head_dim) at the cache's
        current length and returns the full (cached + new) key/value tensors
        for that layer, up to but not including `advance()`'s new length.
        """
        new_length = self._length + k.shape[-2]
        if new_length > self._max_seq_len:
            raise PromptTooLongError(
                f"context length {new_length} exceeds num_ctx={self._max_seq_len}"
            )
        self._k[layer_idx][:, :, self._length : new_length, :] = k
        self._v[layer_idx][:, :, self._length : new_length, :] = v
        return (
            self._k[layer_idx][:, :, :new_length, :],
            self._v[layer_idx][:, :, :new_length, :],
        )

    def advance(self, n_new_tokens: int) -> None:
        self._length += n_new_tokens

    def truncate(self, length: int) -> None:
        """Rolls the committed length back to `length` (<= the current length) - positions past it
        are simply overwritten by the next `update()`, nothing needs clearing."""
        if not 0 <= length <= self._length:
            raise ValueError(f"can't truncate a cache of length {self._length} to {length}")
        self._length = length

    def drop_range(
        self, start: int, count: int, rotations: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Removes positions `[start, start + count)`: everything after slides down by `count` and
        its
        keys are rotated (one `(cos, sin)` per layer, see `RotaryEmbedding.shift_table`) to carry
        the
        new position, so the kept tokens need no recomputing. Values hold no position - just
        moved."""
        end = self._length
        if start < 0 or count <= 0 or start + count > end or len(rotations) != len(self._k):
            raise ValueError(f"can't drop {count} positions at {start} of a cache of length {end}")
        for layer, (cos, sin) in enumerate(rotations):
            keys = self._k[layer][:, :, start + count : end, :]
            moved = keys.float()
            moved = moved * cos.to(moved.device) + rotate_half(moved) * sin.to(moved.device)
            self._k[layer][:, :, start : end - count, :] = moved.to(self._dtype)
            self._v[layer][:, :, start : end - count, :] = self._v[layer][
                :, :, start + count : end, :
            ].clone()
        self._length = end - count

    def fork(self, length: int, max_seq_len: int | None = None) -> "KVCache":
        """A new cache holding a copy of the first `length` positions; this one is left untouched.
        Cheap next to recomputing those positions - how `PromptCache` lets a second conversation
        build on a shared prefix without destroying the first one's cache. `max_seq_len` gives the
        copy a different capacity (a larger `num_ctx` than this cache was built for)."""
        if not 0 <= length <= self._length:
            raise ValueError(f"can't fork a cache of length {self._length} at {length}")
        capacity = max_seq_len or self._max_seq_len
        if capacity < length:
            raise ValueError(f"can't fork {length} positions into a capacity of {capacity}")
        forked = KVCache(self._layer_shapes, capacity, self._dtype, self._device)
        for src, dst in ((self._k, forked._k), (self._v, forked._v)):
            for layer_src, layer_dst in zip(src, dst, strict=True):
                layer_dst[:, :, :length, :] = layer_src[:, :, :length, :]
        forked._length = length
        return forked

    def nbytes(self) -> int:
        """Memory held by the preallocated key/value tensors."""
        return sum(t.numel() * t.element_size() for t in (*self._k, *self._v))

    def export(self, length: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Copies of the first `length` positions' keys and values, one tensor per layer - owning
        only those positions (a bare slice would drag the whole preallocated buffer into a save)."""
        if not 0 <= length <= self._length:
            raise ValueError(f"can't export {length} positions of a cache of length {self._length}")
        return (
            [t[:, :, :length, :].clone() for t in self._k],
            [t[:, :, :length, :].clone() for t in self._v],
        )

    def load(self, keys: list[torch.Tensor], values: list[torch.Tensor]) -> None:
        """Fills an empty cache from `export()` output and sets its length to match. A mismatch
        with this cache's layers/shapes/dtype (a stale or foreign save) raises `ValueError`."""
        if len(keys) != len(self._k) or len(values) != len(self._v):
            raise ValueError("saved cache has a different number of layers")
        length = keys[0].shape[2] if keys else 0
        if length > self._max_seq_len:
            raise ValueError(f"saved cache has {length} positions, num_ctx is {self._max_seq_len}")
        for dst_list, src_list in ((self._k, keys), (self._v, values)):
            for dst, src in zip(dst_list, src_list, strict=True):
                if src.shape != (*dst.shape[:2], length, dst.shape[3]) or src.dtype != dst.dtype:
                    raise ValueError("saved cache layer shape or dtype differs from this model's")
                dst[:, :, :length, :] = src
        self._length = length

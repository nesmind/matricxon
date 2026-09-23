import torch

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
    ) -> None:
        self._k = [
            torch.zeros((1, n_head_kv, max_seq_len, head_dim), dtype=dtype)
            for n_head_kv, head_dim in layer_shapes
        ]
        self._v = [
            torch.zeros((1, n_head_kv, max_seq_len, head_dim), dtype=dtype)
            for n_head_kv, head_dim in layer_shapes
        ]
        self._max_seq_len = max_seq_len
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

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

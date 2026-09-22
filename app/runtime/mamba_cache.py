import torch

from app.runtime.kv_cache import KVCache


class NemotronHHybridCache:
    """Composes two independent, orthogonal sub-caches - one real `KVCache` restricted to this
    model's real attention-layer indices, plus a fixed-size (never grows with sequence length)
    conv-state + ssm-state pair per real Mamba-2 layer index - mirrors llama.cpp's own real
    `llama_memory_hybrid` shape (two independently-typed caches, dispatched per layer index via a
    predicate computed once at load time), reimplemented here rather than copied. Plain-MLP layer
    indices touch neither sub-structure - they carry no persistent state at all.

    Lifetime is a single `ChatEngine.stream()` call, same as plain `KVCache` (see that class's own
    docstring) - no cross-request reuse. `ChatEngine` itself only ever touches `.length`/
    `.advance()` (both delegate to the inner `KVCache`, since both layer kinds process the same
    shared sequence position) - `update_attention`/`mamba_state`/`set_mamba_state` are called only
    by `NemotronHArchitecture`'s own layer classes, so there's no need for this to satisfy any
    generic/polymorphic cache interface beyond what `ChatEngine` itself needs.
    """

    def __init__(
        self,
        layer_types: list[str],
        attention_layer_shape: tuple[int, int],
        mamba_conv_state_shape: tuple[int, int],
        mamba_ssm_state_shape: tuple[int, int, int],
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        attention_indices = [i for i, t in enumerate(layer_types) if t == "attention"]
        mamba_indices = [i for i, t in enumerate(layer_types) if t == "mamba"]
        self._kv_cache = KVCache(
            layer_shapes=[attention_layer_shape] * len(attention_indices),
            max_seq_len=max_seq_len,
            dtype=dtype,
        )
        self._attention_slot = {layer_idx: slot for slot, layer_idx in enumerate(attention_indices)}
        self._mamba_slot = {layer_idx: slot for slot, layer_idx in enumerate(mamba_indices)}
        self.conv_state = [
            torch.zeros(1, *mamba_conv_state_shape, dtype=dtype) for _ in mamba_indices
        ]
        self.ssm_state = [
            torch.zeros(1, *mamba_ssm_state_shape, dtype=dtype) for _ in mamba_indices
        ]

    @property
    def length(self) -> int:
        return self._kv_cache.length

    def advance(self, n_new_tokens: int) -> None:
        self._kv_cache.advance(n_new_tokens)

    def update_attention(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._kv_cache.update(self._attention_slot[layer_idx], k, v)

    def mamba_state(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self._mamba_slot[layer_idx]
        return self.conv_state[slot], self.ssm_state[slot]

    def set_mamba_state(
        self, layer_idx: int, conv_state: torch.Tensor, ssm_state: torch.Tensor
    ) -> None:
        slot = self._mamba_slot[layer_idx]
        self.conv_state[slot] = conv_state
        self.ssm_state[slot] = ssm_state

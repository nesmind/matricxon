from dataclasses import dataclass

import torch

from app.runtime.kv_cache import KVCache


@dataclass(frozen=True)
class HybridSnapshot:
    """The recurrent half of a `NemotronHHybridCache` at one sequence position, plus that position.
    (The attention half is a KV prefix: restoring it is a truncate.)"""

    length: int
    conv_state: list[torch.Tensor]
    ssm_state: list[torch.Tensor]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.conv_state, *self.ssm_state))


class NemotronHHybridCache:
    """Composes two orthogonal sub-caches: a `KVCache` restricted to this model's attention-layer
    indices, plus a fixed-size (not growing with sequence length) conv-state + ssm-state pair per
    Mamba-2 layer index. Mirrors llama.cpp's `llama_memory_hybrid` (two independently typed caches
    dispatched per layer index by a predicate computed at load time), reimplemented, not copied.
    Plain-MLP layers touch neither.

    Lifetime is one `ChatEngine.stream` call, like plain `KVCache`. `ChatEngine` touches only
    `.length`/`.advance` (both delegate to the inner `KVCache`, as both layer kinds process the same
    position); `update_attention`/`mamba_state`/`set_mamba_state` are called only by
    `NemotronHArchitecture`'s layers, so no generic cache interface is needed beyond `ChatEngine`'s.
    """

    def __init__(
        self,
        layer_types: list[str],
        attention_layer_shape: tuple[int, int],
        mamba_conv_state_shape: tuple[int, int],
        mamba_ssm_state_shape: tuple[int, int, int],
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self._spec = (
            layer_types,
            attention_layer_shape,
            mamba_conv_state_shape,
            mamba_ssm_state_shape,
            max_seq_len,
            dtype,
            device,
        )
        attention_indices = [i for i, t in enumerate(layer_types) if t == "attention"]
        mamba_indices = [i for i, t in enumerate(layer_types) if t == "mamba"]
        self._kv_cache = KVCache(
            layer_shapes=[attention_layer_shape] * len(attention_indices),
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )
        self._attention_slot = {layer_idx: slot for slot, layer_idx in enumerate(attention_indices)}
        self._mamba_slot = {layer_idx: slot for slot, layer_idx in enumerate(mamba_indices)}
        self.conv_state = [
            torch.zeros(1, *mamba_conv_state_shape, dtype=dtype, device=device)
            for _ in mamba_indices
        ]
        self.ssm_state = [
            torch.zeros(1, *mamba_ssm_state_shape, dtype=dtype, device=device)
            for _ in mamba_indices
        ]

    @property
    def length(self) -> int:
        return self._kv_cache.length

    @property
    def device(self) -> torch.device:
        return self._kv_cache.device

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

    def snapshot(self) -> HybridSnapshot:
        """Copies the recurrent state at the current length: the one thing a recurrent layer can't
        be rolled back to otherwise, which `PromptCache` needs to reuse a prefix."""
        return HybridSnapshot(
            self.length,
            [t.clone() for t in self.conv_state],
            [t.clone() for t in self.ssm_state],
        )

    def restore(self, snapshot: HybridSnapshot) -> None:
        """Rolls back to `snapshot` (taken at a length <= the current): the attention KV is
        truncated, the recurrent state replaced by a copy (the snapshot stays reusable)."""
        self._kv_cache.truncate(snapshot.length)
        self.conv_state = [t.clone() for t in snapshot.conv_state]
        self.ssm_state = [t.clone() for t in snapshot.ssm_state]

    def fork_from(
        self, snapshot: HybridSnapshot, max_seq_len: int | None = None
    ) -> "NemotronHHybridCache":
        """A new cache at `snapshot.length`: a copy of this one's attention KV up to there plus the
        snapshot's recurrent state; this cache is untouched (`KVCache.fork`, which also explains
        `max_seq_len`)."""
        spec = list(self._spec)
        spec[4] = max_seq_len or spec[4]
        forked = NemotronHHybridCache(*spec)
        forked._kv_cache = self._kv_cache.fork(snapshot.length, max_seq_len)
        forked.conv_state = [t.clone() for t in snapshot.conv_state]
        forked.ssm_state = [t.clone() for t in snapshot.ssm_state]
        return forked

    def nbytes(self) -> int:
        """Memory held: the attention KV plus the live recurrent state."""
        states = (*self.conv_state, *self.ssm_state)
        return self._kv_cache.nbytes() + sum(t.numel() * t.element_size() for t in states)

    def export(self, length: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The attention KV's first `length` positions (`KVCache.export`). The recurrent state isn't
        included: it is meaningful only as a `HybridSnapshot`."""
        return self._kv_cache.export(length)

    def load(self, keys: list[torch.Tensor], values: list[torch.Tensor]) -> None:
        """Fills the attention KV from `export` output; the recurrent state stays zero until a
        snapshot is restored into it (`restore`), the only way a saved cache is used."""
        self._kv_cache.load(keys, values)

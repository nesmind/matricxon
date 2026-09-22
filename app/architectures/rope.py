import math

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Plain (unscaled) rotary position embedding: `inv_freq = 1/theta^(2i/dim)`,
    no long-context correction. Used by `nomic-bert` - its GGUF metadata
    carries only `rope.freq_base`, none of YaRN's `rope.scaling.*` keys
    (confirmed against the real `nomic-embed-text-v1.5` GGUF) - `YarnRotaryEmbedding`
    below extends this with that correction for architectures that need it.
    """

    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_factor = 1.0

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos() * self.attention_factor, emb.sin() * self.attention_factor


class YarnRotaryEmbedding(RotaryEmbedding):
    """YaRN-scaled rotary position embedding.

    Ports HuggingFace transformers' `_compute_yarn_parameters`
    (modeling_rope_utils.py), verified against that source rather than
    re-derived from memory - a wrong correction-range or ramp here would
    silently produce garbage generations instead of an obvious error.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        factor: float,
        beta_fast: float,
        beta_slow: float,
        original_context_length: int,
        mscale: float | None,
        mscale_all_dim: float | None,
    ) -> None:
        nn.Module.__init__(self)  # skip RotaryEmbedding.__init__ - we compute our own inv_freq
        inv_freq, attention_factor = self._compute(
            head_dim,
            rope_theta,
            factor,
            beta_fast,
            beta_slow,
            original_context_length,
            mscale,
            mscale_all_dim,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_factor = attention_factor

    @staticmethod
    def _get_mscale(scale: float, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def _find_correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
        return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    @classmethod
    def _find_correction_range(
        cls, low_rot: float, high_rot: float, dim: int, base: float, max_pos: int
    ) -> tuple[float, float]:
        low = math.floor(cls._find_correction_dim(low_rot, dim, base, max_pos))
        high = math.ceil(cls._find_correction_dim(high_rot, dim, base, max_pos))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _linear_ramp_factor(low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high += 0.001
        linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear, 0, 1)

    def _compute(
        self,
        dim: int,
        base: float,
        factor: float,
        beta_fast: float,
        beta_slow: float,
        original_ctx: int,
        mscale: float | None,
        mscale_all_dim: float | None,
    ) -> tuple[torch.Tensor, float]:
        if mscale is not None and mscale_all_dim is not None:
            attention_factor = self._get_mscale(factor, mscale) / self._get_mscale(
                factor, mscale_all_dim
            )
        else:
            attention_factor = self._get_mscale(factor)

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = self._find_correction_range(beta_fast, beta_slow, dim, base, original_ctx)
        extrap_factor = 1 - self._linear_ramp_factor(low, high, dim // 2)
        inv_freq = (
            inv_freq_interpolation * (1 - extrap_factor) + inv_freq_extrapolation * extrap_factor
        )
        return inv_freq, attention_factor


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_partial(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NEOX-style *partial* rotary: only the first `cos.shape[-1]` dims of each head are rotated,
    the rest pass through unchanged - real Phi-2 (`rope.dimension_count=32` of a real 64-dim head,
    confirmed against a real moondream2 GGUF pull, not assumed) - every other architecture here
    rotates the whole head (`cos`/`sin` already sized to the full `head_dim`), so this is additive:
    `apply_rotary_pos_emb`/`rotate_half` above are unchanged and every existing caller is
    unaffected. `cos`/`sin` come from a `RotaryEmbedding` constructed with `head_dim=rot_dim` (not
    the real full head_dim) - `inv_freq` only ever depends on the width being rotated, so no
    changes to `RotaryEmbedding` itself are needed to produce them.
    """
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

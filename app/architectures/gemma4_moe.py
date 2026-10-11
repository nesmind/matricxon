"""Gemma4's Mixture-of-Experts block, different from GraniteMoE/Mixtral's
(`granitemoe_layers.py`/`llama_moe.py`), so not reusable unchanged. Ported from HF `transformers`
(`models/gemma4/modular_gemma4.py`'s `Gemma4TextRouter`/`Gemma4TextDecoderLayer`,
`Gemma4TextExperts(MixtralExperts)` -> `models/mixtral/modeling_mixtral.py`) and cross-checked
against llama.cpp (`src/models/gemma4.cpp`); both agree.

Differences from GraniteMoeFFN:
- The router runs an unscaled RMSNorm, then a learned elementwise `scale` + a fixed
  `1/sqrt(hidden_size)` multiply before its projection (GraniteMoE's router is a bare `nn.Linear` on
  the raw hidden state).
- A learned `per_expert_scale` multiplies the final top-k weights after renormalization (GraniteMoE
  has none).
- The expert FFN uses Gemma4's `gelu_pytorch_tanh` (HF `config.hidden_activation`, as its dense
  `Gemma4MLP`), not SiLU.
- The result is added to an always-computed shared dense MLP branch (each with its own extra norm),
  not replacing the FFN; GraniteMoE has no shared expert.
Router order (softmax over all experts, topk, renormalize) equals GraniteMoE's topk-then-softmax
(proved when that class was reused for Mixtral; `llama_moe.py`): softmax is monotonic, so both pick
the same top-k, and renormalizing a softmax subset equals softmax over just it.

No real gemma-4-*-A*B-it (MoE) GGUF exists to validate against: the only installed Gemma4 checkpoint
(`gemma-4-E2B-it`) is dense (`expert_count=0`, `Gemma4Architecture.unsupported_features`'s `"moe"`
flag). A real-weight validation gap in ROADMAP.md, like command-r/falcon.
"""

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.gemma4_layers import rms_norm_no_scale
from app.architectures.layers import RMSNorm
from app.architectures.moe_experts import QuantizedMoEExperts, materialize_quantized_moe_experts
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata

_GELU_TANH = partial(F.gelu, approximate="tanh")


def detect_moe(metadata: GGUFMetadata, arch: str) -> tuple[int | None, int | None, int | None]:
    """(num_experts, num_experts_per_tok, expert_ffn_len), all None for a dense gemma4 file. Same
    `expert_count`/
    `expert_used_count` keys as `llama_moe.detect_moe`/`GraniteMoeArchitecture`, plus gemma4's own
    `expert_feed_forward_length` (independent of the shared dense MLP's `ffn_len`; llama.cpp reads
    it via
    `LLM_KV_EXPERT_FEED_FORWARD_LENGTH`, not `feed_forward_length`).
    """
    num_experts = metadata.get_u32(f"{arch}.expert_count")
    if not num_experts:
        return None, None, None
    return (
        num_experts,
        metadata.get_u32(f"{arch}.expert_used_count"),
        metadata.get_u32(f"{arch}.expert_feed_forward_length"),
    )


class Gemma4Router(nn.Module):
    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.rms_eps = rms_eps
        self.proj = nn.Linear(n_embd, num_experts, bias=False, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(n_embd, dtype=dtype))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts, dtype=dtype))
        self._scalar_root_size = n_embd**-0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = rms_norm_no_scale(x, self.rms_eps) * self.scale * self._scalar_root_size
        probs = F.softmax(self.proj(x).float(), dim=-1)
        top_k_weights, top_k_idx = probs.topk(self.num_experts_per_tok, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_idx]
        return top_k_weights.to(x.dtype), top_k_idx


class Gemma4MoEBlock(nn.Module):
    """Combines an always-computed dense MLP output with a sparse expert branch by addition (op
    order from HF
    `modular_gemma4.py` and llama.cpp's `src/models/gemma4.cpp`, which agree):
    `post_norm_1(dense_out) + post_norm_2(experts(router(pre_norm_2(residual))))`. Router/experts
    see `residual`
    (the pre-dense-MLP hidden state), not the dense MLP's output.
    """

    def __init__(
        self,
        n_embd: int,
        ffn_len: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.router = Gemma4Router(n_embd, num_experts, num_experts_per_tok, rms_eps, dtype=dtype)
        self.experts = QuantizedMoEExperts(n_embd, ffn_len, num_experts, _GELU_TANH, dtype=dtype)
        self.post_norm_1 = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.pre_norm_2 = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.post_norm_2 = RMSNorm(n_embd, rms_eps, dtype=dtype)

    def forward(self, dense_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = residual.shape
        hidden_states_1 = self.post_norm_1(dense_out)

        hidden_states_2 = self.pre_norm_2(residual).reshape(seq_len, n_embd)
        top_k_weights, top_k_idx = self.router(hidden_states_2)
        hidden_states_2 = self.experts(hidden_states_2, top_k_weights, top_k_idx)
        hidden_states_2 = self.post_norm_2(hidden_states_2.reshape(batch, seq_len, n_embd))

        return hidden_states_1 + hidden_states_2


def materialize_moe(
    moe: Gemma4MoEBlock,
    loader: GGUFModelLoader,
    prefix: str,
    enabled: bool,
    mark_used: Callable[[GGUFModelLoader], None],
) -> None:
    """Router/norms: a plain `.copy_` (small per-model/per-expert scalars and vectors; packing
    wouldn't help).
    Experts: `QuantizedMoEExperts` (`moe_experts.py`), per-expert `QuantizedLinear`s when `enabled`
    and a GEMV kernel exists for this checkpoint's expert-tensor type.
    """
    moe.router.proj.weight.copy_(loader.load_tensor(prefix + "ffn_gate_inp.weight"))
    moe.router.scale.data.copy_(loader.load_tensor(prefix + "ffn_gate_inp.scale"))
    moe.router.per_expert_scale.data.copy_(loader.load_tensor(prefix + "ffn_down_exps.scale"))
    materialize_quantized_moe_experts(moe.experts, loader, prefix, enabled, mark_used)
    moe.post_norm_1.weight.copy_(loader.load_tensor(prefix + "ffn_post_norm_1.weight"))
    moe.pre_norm_2.weight.copy_(loader.load_tensor(prefix + "ffn_pre_norm_2.weight"))
    moe.post_norm_2.weight.copy_(loader.load_tensor(prefix + "ffn_post_norm_2.weight"))

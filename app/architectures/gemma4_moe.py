"""Gemma4's real Mixture-of-Experts block - genuinely different from GraniteMoE/Mixtral's own
MoE (both already implemented, see `granitemoe_layers.py`/`llama_moe.py`), not reusable
unchanged. Ported from real HF `transformers` source (`transformers/models/gemma4/
modular_gemma4.py`'s `Gemma4TextRouter`/`Gemma4TextDecoderLayer`,
`Gemma4TextExperts(MixtralExperts)` -> `transformers/models/mixtral/modeling_mixtral.py`'s real
`MixtralExperts`) and cross-checked against real llama.cpp GGUF-inference source
(`src/models/gemma4.cpp`) - both agree.

Real, confirmed differences from GraniteMoeFFN's own MoE:
- The router runs an unscaled RMSNorm, then a learned elementwise `scale` + a fixed
  `1/sqrt(hidden_size)` multiply, *before* its projection - GraniteMoE's router is a bare
  `nn.Linear` on the raw hidden state, no pre-norm/pre-scale at all.
- A learned `per_expert_scale` multiplies the final top-k weights *after* renormalization -
  GraniteMoE has no such per-expert term.
- The expert FFN uses Gemma4's own real `gelu_pytorch_tanh` activation (real HF
  `config.hidden_activation`, same as its dense `Gemma4MLP`), not GraniteMoE's hardcoded SiLU.
- The result is *added* to a real, always-computed shared dense MLP branch (each wrapped in its
  own extra norm) rather than replacing the FFN outright - GraniteMoE has no shared expert at
  all (see that class's own docstring).
Router order (softmax-over-all-experts-then-topk-then-renormalize) is still algebraically
identical to GraniteMoE's topk-then-softmax, for the same reason already proven when that class
was reused for Mixtral (see `llama_moe.py`'s own docstring) - softmax is monotonic, so both pick
the same top-k set, and renormalizing a softmax subset reproduces softmax computed over just it.

No real gemma-4-*-A*B-it (MoE) GGUF exists to validate this against yet - the only real Gemma4
checkpoint installed anywhere in this project (`gemma-4-E2B-it`) is dense (`expert_count=0`, see
`Gemma4Architecture.unsupported_features`'s own `"moe"` flag) - tracked as a real-weight
validation gap in ROADMAP.md, same category as command-r/falcon's own unvalidated real-weight
generation.
"""

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.gemma4_layers import rms_norm_no_scale
from app.architectures.layers import RMSNorm
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata


def detect_moe(metadata: GGUFMetadata, arch: str) -> tuple[int | None, int | None, int | None]:
    """(num_experts, num_experts_per_tok, expert_ffn_len), all None for a dense (non-MoE) real
    gemma4 file - same real `expert_count`/`expert_used_count` keys `llama_moe.detect_moe`/
    `GraniteMoeArchitecture` already read, plus gemma4's own separate `expert_feed_forward_length`
    (its expert FFN width is real and independent of the shared dense MLP's own `ffn_len`,
    confirmed: real llama.cpp reads it via a distinct `LLM_KV_EXPERT_FEED_FORWARD_LENGTH` key,
    not reusing `feed_forward_length`)."""
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


class Gemma4Experts(nn.Module):
    """Same sparse per-expert dispatch as `GraniteMoeFFN`'s own expert loop (gather routed
    tokens, apply that expert, scale, scatter back) - see that class's own docstring for why
    sparse dispatch over dense-then-mask, still true here."""

    def __init__(
        self, n_embd: int, ffn_len: int, num_experts: int, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.gate_exps = nn.Parameter(torch.empty(num_experts, ffn_len, n_embd, dtype=dtype))
        self.up_exps = nn.Parameter(torch.empty(num_experts, ffn_len, n_embd, dtype=dtype))
        self.down_exps = nn.Parameter(torch.empty(num_experts, n_embd, ffn_len, dtype=dtype))

    def forward(
        self, x: torch.Tensor, top_k_weights: torch.Tensor, top_k_idx: torch.Tensor
    ) -> torch.Tensor:
        seq_len, n_embd = x.shape
        out = torch.zeros(seq_len, n_embd, dtype=torch.float32)
        for expert_id in top_k_idx.unique().tolist():
            token_idx, k_idx = (top_k_idx == expert_id).nonzero(as_tuple=True)
            x_e = x.index_select(0, token_idx)
            gate = F.gelu(x_e @ self.gate_exps[expert_id].T, approximate="tanh")
            up = x_e @ self.up_exps[expert_id].T
            down = (gate * up) @ self.down_exps[expert_id].T
            weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, (down * weight).to(torch.float32))
        return out.to(x.dtype)


class Gemma4MoEBlock(nn.Module):
    """Combines a real, always-computed dense MLP output with a sparse expert branch by
    addition - real op order (confirmed from both HF `modular_gemma4.py` and llama.cpp's
    `src/models/gemma4.cpp`, they agree): `post_norm_1(dense_out) + post_norm_2(experts(
    router(pre_norm_2(residual))))`. The router/experts see `residual` (the pre-dense-MLP
    hidden state), not the dense MLP's own output."""

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
        self.experts = Gemma4Experts(n_embd, ffn_len, num_experts, dtype=dtype)
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


def materialize_moe(moe: Gemma4MoEBlock, loader: GGUFModelLoader, prefix: str) -> None:
    """MoE tensors: always a plain `.copy_()`, same reasoning as `llama_moe.materialize_moe_ffn`
    - a real 3D per-expert tensor never goes through `_load_projection`."""
    moe.router.proj.weight.copy_(loader.load_tensor(prefix + "ffn_gate_inp.weight"))
    moe.router.scale.data.copy_(loader.load_tensor(prefix + "ffn_gate_inp.scale"))
    moe.router.per_expert_scale.data.copy_(loader.load_tensor(prefix + "ffn_down_exps.scale"))
    moe.experts.gate_exps.copy_(loader.load_tensor(prefix + "ffn_gate_exps.weight"))
    moe.experts.up_exps.copy_(loader.load_tensor(prefix + "ffn_up_exps.weight"))
    moe.experts.down_exps.copy_(loader.load_tensor(prefix + "ffn_down_exps.weight"))
    moe.post_norm_1.weight.copy_(loader.load_tensor(prefix + "ffn_post_norm_1.weight"))
    moe.pre_norm_2.weight.copy_(loader.load_tensor(prefix + "ffn_pre_norm_2.weight"))
    moe.post_norm_2.weight.copy_(loader.load_tensor(prefix + "ffn_post_norm_2.weight"))

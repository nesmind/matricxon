import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.moe_experts import QuantizedMoEExperts


class GraniteMoeFFN(nn.Module):
    """GraniteMoE's real sparse Mixture-of-Experts FFN block, replacing dense Granite's
    `SwiGLUMLP` inside a `GraniteDecoderLayer` (see that class's own docstring for how it's
    injected) - confirmed against HF `transformers`' real `GraniteMoeTopKRouter`/expert dispatch
    (`modeling_granitemoe.py`, 2026-09-22):

    Router: a plain, bias-free linear projection to `num_experts` logits
    (`blk.N.ffn_gate_inp.weight`, real GGUF shape `(num_experts, hidden_size)`), then
    **top-k THEN softmax** - `topk` selects the `num_experts_per_tok` highest logits *first*,
    and softmax is applied only over those `k` selected logits (renormalizing weights among just
    the selected experts). This is the opposite order from softmax-over-all-experts-then-select,
    an easy, real mistake to make and get numerically wrong in a way that still "runs."

    Experts: `QuantizedMoEExperts` (`app/architectures/moe_experts.py`) - a standard SwiGLU FFN
    (`down(silu(gate(x)) * up(x))`), real 3D per-projection tensors (llama.cpp's Mixtral-style
    MoE convention), quantized-native-aware. No shared/always-on expert - the real small
    checkpoints this was built against (`granite-3.0-1b-a400m-instruct`,
    `granite-3.0-3b-a800m-instruct`) both use the plain `GraniteMoeForCausalLM` HF class,
    confirmed via live `config.json` to have no `shared_intermediate_size` field (that's a
    separate `GraniteMoeSharedForCausalLM` HF class, out of scope here). Reused unchanged for
    Mixtral (see `llama_moe.py`'s own docstring for the proven router-math equivalence).
    """

    def __init__(
        self,
        n_embd: int,
        ffn_len: int,
        num_experts: int,
        num_experts_per_tok: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.router = nn.Linear(n_embd, num_experts, bias=False, dtype=dtype)
        self.experts = QuantizedMoEExperts(n_embd, ffn_len, num_experts, F.silu, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape
        x_flat = x.reshape(seq_len, n_embd)

        router_logits = self.router(x_flat)
        top_k_logits, top_k_idx = router_logits.topk(self.num_experts_per_tok, dim=-1)
        # float32 softmax for the same numerical-stability reason RMSNorm computes in float32
        # regardless of the surrounding activation dtype.
        top_k_weights = F.softmax(top_k_logits.float(), dim=-1).to(x.dtype)

        out = self.experts(x_flat, top_k_weights, top_k_idx)
        return out.reshape(batch, seq_len, n_embd)

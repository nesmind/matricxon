import torch
import torch.nn.functional as F
from torch import nn


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

    Experts: each is a standard SwiGLU FFN (`down(silu(gate(x)) * up(x))`), stored as one 3D
    tensor per projection type (llama.cpp's Mixtral-style MoE convention) rather than N separate
    2D tensors - `blk.N.ffn_gate_exps.weight`/`ffn_up_exps.weight` shape `(num_experts, ffn_dim,
    hidden_size)`, `ffn_down_exps.weight` shape `(num_experts, hidden_size, ffn_dim)`. No shared/
    always-on expert - the real small checkpoints this was built against
    (`granite-3.0-1b-a400m-instruct`, `granite-3.0-3b-a800m-instruct`) both use the plain
    `GraniteMoeForCausalLM` HF class, confirmed via live `config.json` to have no
    `shared_intermediate_size` field (that's a separate `GraniteMoeSharedForCausalLM` HF class,
    out of scope here).

    Forward pass uses **sparse dispatch** (loop only over the experts actually selected this
    call, never all `num_experts`) rather than densely computing every expert for every token
    then masking - matricxon's real call shape is always `batch == 1` (either prefill
    `seq_len > 1` or decode `seq_len == 1`, confirmed project-wide invariant, see
    `QuantizedLinear`'s own docstring), and the dominant real workload is decode - one token at a
    time - where sparse dispatch does exactly `num_experts_per_tok` expert matmuls instead of
    `num_experts`, not a micro-optimization (e.g. 8 vs 32 for `granite-3.0-1b-a400m-instruct`).
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
        self.gate_exps = nn.Parameter(torch.empty(num_experts, ffn_len, n_embd, dtype=dtype))
        self.up_exps = nn.Parameter(torch.empty(num_experts, ffn_len, n_embd, dtype=dtype))
        self.down_exps = nn.Parameter(torch.empty(num_experts, n_embd, ffn_len, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape
        x_flat = x.reshape(seq_len, n_embd)

        router_logits = self.router(x_flat)
        top_k_logits, top_k_idx = router_logits.topk(self.num_experts_per_tok, dim=-1)
        # float32 softmax for the same numerical-stability reason RMSNorm computes in float32
        # regardless of the surrounding activation dtype.
        top_k_weights = F.softmax(top_k_logits.float(), dim=-1).to(x.dtype)

        # Accumulate in float32 regardless of `x`'s own dtype - `index_add_` summing up to
        # `num_experts_per_tok` per-expert contributions in a lower-precision dtype (e.g. bf16)
        # is a real, plausible source of extra rounding error beyond ordinary quantization noise;
        # cheap to avoid by accumulating wide and casting back once at the end.
        out = torch.zeros(seq_len, n_embd, dtype=torch.float32)
        for expert_id in top_k_idx.unique().tolist():
            token_idx, k_idx = (top_k_idx == expert_id).nonzero(as_tuple=True)
            x_e = x_flat.index_select(0, token_idx)
            gate = F.silu(x_e @ self.gate_exps[expert_id].T)
            up = x_e @ self.up_exps[expert_id].T
            down = (gate * up) @ self.down_exps[expert_id].T
            weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, (down * weight).to(torch.float32))

        return out.to(x.dtype).reshape(batch, seq_len, n_embd)

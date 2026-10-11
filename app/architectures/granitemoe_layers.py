import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.moe_experts import QuantizedMoEExperts


class GraniteMoeFFN(nn.Module):
    """GraniteMoE's sparse MoE FFN, replacing dense Granite's `SwiGLUMLP` in a
    `GraniteDecoderLayer`. Checked against HF `GraniteMoeTopKRouter`/expert dispatch
    (`modeling_granitemoe.py`).

    Router: a bias-free linear to `num_experts` logits (`blk.N.ffn_gate_inp.weight`), then top-k
    THEN softmax: `topk` picks the `num_experts_per_tok` highest logits first and softmax runs over
    only those `k`. Softmax-over-all-then-select would still run but be numerically wrong.

    Experts: `QuantizedMoEExperts` (`moe_experts.py`), a SwiGLU FFN (`down(silu(gate(x)) * up(x))`)
    with 3D per-projection tensors (llama.cpp's Mixtral-style layout), quantized-native-aware. No
    shared expert: the targeted checkpoints (`granite-3.0-1b-a400m-instruct`,
    `granite-3.0-3b-a800m-instruct`) use plain `GraniteMoeForCausalLM` (the `GraniteMoeShared`
    variant is out of scope). Reused unchanged for Mixtral (`llama_moe.py`).
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
        # float32 softmax, for the same stability reason RMSNorm computes in float32 whatever the
        # activation dtype.
        top_k_weights = F.softmax(top_k_logits.float(), dim=-1).to(x.dtype)

        out = self.experts(x_flat, top_k_weights, top_k_idx)
        return out.reshape(batch, seq_len, n_embd)

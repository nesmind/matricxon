"""Standalone correctness tests for `GraniteMoeFFN` (`app/architectures/granitemoe_layers.py`):
isolates the router ordering (top-k THEN softmax) and 3D-expert-tensor indexing from GGUF loading,
tied embeddings, RoPE etc. before it is wired into an architecture, per the risk the module
docstring names."""

import torch
import torch.nn.functional as F

from app.architectures.granitemoe_layers import GraniteMoeFFN

N_EMBD = 6
FFN_LEN = 5


def _expert_forward(
    x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor
) -> torch.Tensor:
    """Independent, deliberately simple reimplementation of one expert's SwiGLU FFN: the reference
    each test checks `GraniteMoeFFN`'s forward against."""
    gate = F.silu(x @ gate_w.T)
    up = x @ up_w.T
    return (gate * up) @ down_w.T


def _random_ffn(num_experts: int, num_experts_per_tok: int, seed: int = 0) -> GraniteMoeFFN:
    torch.manual_seed(seed)
    ffn = GraniteMoeFFN(N_EMBD, FFN_LEN, num_experts, num_experts_per_tok, dtype=torch.float32)
    with torch.no_grad():
        ffn.router.weight.copy_(torch.randn_like(ffn.router.weight))
        ffn.experts.gate_exps.copy_(torch.randn_like(ffn.experts.gate_exps))
        ffn.experts.up_exps.copy_(torch.randn_like(ffn.experts.up_exps))
        ffn.experts.down_exps.copy_(torch.randn_like(ffn.experts.down_exps))
    return ffn


class TestSparseDispatchMatchesBruteForcePerToken:
    def test_multi_token_input_matches_a_naive_per_token_reference(self) -> None:
        num_experts, num_experts_per_tok, seq_len = 6, 2, 5
        ffn = _random_ffn(num_experts, num_experts_per_tok)
        x = torch.randn(1, seq_len, N_EMBD)

        actual = ffn(x)

        # Deliberately slow and independent: token by token, expert by expert, no batching or
        # index_select/index_add, nothing shared with the real code path.
        expected = torch.zeros(seq_len, N_EMBD)
        router_logits = x.reshape(seq_len, N_EMBD) @ ffn.router.weight.T
        for t in range(seq_len):
            top_k_logits, top_k_idx = router_logits[t].topk(num_experts_per_tok)
            weights = F.softmax(top_k_logits, dim=-1)
            for k in range(num_experts_per_tok):
                e = top_k_idx[k].item()
                out_e = _expert_forward(
                    x[0, t : t + 1],
                    ffn.experts.gate_exps[e],
                    ffn.experts.up_exps[e],
                    ffn.experts.down_exps[e],
                )
                expected[t] += weights[k] * out_e[0]

        assert torch.allclose(actual[0], expected, atol=1e-5)

    def test_single_decode_token_matches_the_reference(self) -> None:
        """The dominant workload shape (GraniteMoeFFN docstring): seq_len == 1."""
        num_experts, num_experts_per_tok = 8, 3
        ffn = _random_ffn(num_experts, num_experts_per_tok)
        x = torch.randn(1, 1, N_EMBD)

        actual = ffn(x)

        router_logits = x.reshape(1, N_EMBD) @ ffn.router.weight.T
        top_k_logits, top_k_idx = router_logits[0].topk(num_experts_per_tok)
        weights = F.softmax(top_k_logits, dim=-1)
        expected = torch.zeros(1, N_EMBD)
        for k in range(num_experts_per_tok):
            e = top_k_idx[k].item()
            out_e = _expert_forward(
                x[0], ffn.experts.gate_exps[e], ffn.experts.up_exps[e], ffn.experts.down_exps[e]
            )
            expected += weights[k] * out_e

        assert torch.allclose(actual[0], expected, atol=1e-5)


class TestTopKThenSoftmaxOrder:
    def test_every_expert_selected_matches_a_plain_dense_weighted_sum(self) -> None:
        """Degenerate but easy to verify: num_experts_per_tok == num_experts selects every expert
        for every token, so top-k-then-softmax must reduce to a softmax-over-all-experts weighted
        sum. This alone doesn't discriminate it from softmax-then-topk (nothing is sliced away); the
        brute-force per-token tests above do, since real selection happens there."""
        num_experts = 4
        ffn = _random_ffn(num_experts, num_experts_per_tok=num_experts)
        x = torch.randn(1, 3, N_EMBD)

        actual = ffn(x)

        x_flat = x.reshape(3, N_EMBD)
        router_logits = x_flat @ ffn.router.weight.T
        weights = F.softmax(router_logits, dim=-1)
        expected = torch.zeros(3, N_EMBD)
        for e in range(num_experts):
            out_e = _expert_forward(
                x_flat, ffn.experts.gate_exps[e], ffn.experts.up_exps[e], ffn.experts.down_exps[e]
            )
            expected += weights[:, e : e + 1] * out_e

        assert torch.allclose(actual[0], expected, atol=1e-5)

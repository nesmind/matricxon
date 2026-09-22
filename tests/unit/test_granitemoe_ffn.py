"""Standalone correctness tests for `GraniteMoeFFN` (`app/architectures/granitemoe_layers.py`) -
isolates the real router-ordering (top-k THEN softmax) and 3D-expert-tensor-indexing logic from
every other moving part (GGUF loading, tied embeddings, RoPE, ...) before it's wired into a full
architecture, per the real risk this module's own docstring calls out.
"""

import torch
import torch.nn.functional as F

from app.architectures.granitemoe_layers import GraniteMoeFFN

N_EMBD = 6
FFN_LEN = 5


def _expert_forward(
    x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor
) -> torch.Tensor:
    """Independent, deliberately-simple reimplementation of one expert's SwiGLU FFN - the
    reference every test below checks `GraniteMoeFFN`'s real forward pass against."""
    gate = F.silu(x @ gate_w.T)
    up = x @ up_w.T
    return (gate * up) @ down_w.T


def _random_ffn(num_experts: int, num_experts_per_tok: int, seed: int = 0) -> GraniteMoeFFN:
    torch.manual_seed(seed)
    ffn = GraniteMoeFFN(N_EMBD, FFN_LEN, num_experts, num_experts_per_tok, dtype=torch.float32)
    with torch.no_grad():
        ffn.router.weight.copy_(torch.randn_like(ffn.router.weight))
        ffn.gate_exps.copy_(torch.randn_like(ffn.gate_exps))
        ffn.up_exps.copy_(torch.randn_like(ffn.up_exps))
        ffn.down_exps.copy_(torch.randn_like(ffn.down_exps))
    return ffn


class TestSparseDispatchMatchesBruteForcePerToken:
    def test_multi_token_input_matches_a_naive_per_token_reference(self) -> None:
        num_experts, num_experts_per_tok, seq_len = 6, 2, 5
        ffn = _random_ffn(num_experts, num_experts_per_tok)
        x = torch.randn(1, seq_len, N_EMBD)

        actual = ffn(x)

        # Deliberately slow, deliberately independent: loop token by token, expert by expert -
        # no batching, no index_select/index_add, nothing shared with the real implementation's
        # own code path.
        expected = torch.zeros(seq_len, N_EMBD)
        router_logits = x.reshape(seq_len, N_EMBD) @ ffn.router.weight.T
        for t in range(seq_len):
            top_k_logits, top_k_idx = router_logits[t].topk(num_experts_per_tok)
            weights = F.softmax(top_k_logits, dim=-1)
            for k in range(num_experts_per_tok):
                e = top_k_idx[k].item()
                out_e = _expert_forward(
                    x[0, t : t + 1], ffn.gate_exps[e], ffn.up_exps[e], ffn.down_exps[e]
                )
                expected[t] += weights[k] * out_e[0]

        assert torch.allclose(actual[0], expected, atol=1e-5)

    def test_single_decode_token_matches_the_reference(self) -> None:
        """The dominant real workload shape (see GraniteMoeFFN's own docstring) - seq_len == 1."""
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
            out_e = _expert_forward(x[0], ffn.gate_exps[e], ffn.up_exps[e], ffn.down_exps[e])
            expected += weights[k] * out_e

        assert torch.allclose(actual[0], expected, atol=1e-5)


class TestTopKThenSoftmaxOrder:
    def test_every_expert_selected_matches_a_plain_dense_weighted_sum(self) -> None:
        """Degenerate but easy-to-verify case: num_experts_per_tok == num_experts means every
        expert is "selected" for every token, so top-k-then-softmax must reduce to an ordinary
        softmax-over-all-experts weighted sum - the real invariant that pins top-k-then-softmax
        (not softmax-then-topk) is the correct order: had the module instead done softmax over
        all experts and only *then* sliced out non-selected ones, this case would still
        accidentally match (nothing is sliced away), so this test alone doesn't fully
        discriminate the two orders - it's paired with the brute-force per-token tests above,
        which do, since real selection actually happens there.
        """
        num_experts = 4
        ffn = _random_ffn(num_experts, num_experts_per_tok=num_experts)
        x = torch.randn(1, 3, N_EMBD)

        actual = ffn(x)

        x_flat = x.reshape(3, N_EMBD)
        router_logits = x_flat @ ffn.router.weight.T
        weights = F.softmax(router_logits, dim=-1)  # (T, num_experts)
        expected = torch.zeros(3, N_EMBD)
        for e in range(num_experts):
            out_e = _expert_forward(x_flat, ffn.gate_exps[e], ffn.up_exps[e], ffn.down_exps[e])
            expected += weights[:, e : e + 1] * out_e

        assert torch.allclose(actual[0], expected, atol=1e-5)

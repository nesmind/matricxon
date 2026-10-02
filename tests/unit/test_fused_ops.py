"""The fused decode ops (RMSNorm, RoPE, cached attention, the sampler's draw) must equal the torch
code they replace, on BOTH backends: the native C library and the Numba twins."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from app.architectures.layers import RMSNorm
from app.architectures.rope import rotate_half
from app.native.fused_ops import FusedOps, _NativeBackend, _NumbaBackend
from app.native.library import NativeKernelLibrary
from app.runtime.generation_request import SamplingConfig
from app.runtime.sampler import RepetitionPenaltyFilter, TopKFilter, TopPFilter


@pytest.fixture(scope="module", params=["native", "numba"])
def ops(request: pytest.FixtureRequest) -> FusedOps:
    if request.param == "native":
        return FusedOps(_NativeBackend(NativeKernelLibrary().load(), 2))
    return FusedOps(_NumbaBackend(attention=True))


class TestRmsNorm:
    @pytest.mark.parametrize("shape", [(1, 1, 96), (1, 7, 64), (3, 128)])
    def test_matches_the_module(self, ops: FusedOps, shape: tuple[int, ...]) -> None:
        torch.manual_seed(0)
        norm = RMSNorm(shape[-1], 1e-5)
        norm.weight.data = torch.randn(shape[-1])
        x = torch.randn(*shape)
        fused = ops.rms_norm(x, norm.weight, norm.eps)
        assert fused is not None and fused.shape == x.shape
        assert torch.allclose(fused, norm(x), atol=1e-5, rtol=1e-5)

    def test_declines_what_it_cannot_do(self, ops: FusedOps) -> None:
        w = torch.ones(8)
        assert ops.rms_norm(torch.randn(1, 8, dtype=torch.bfloat16), w, 1e-5) is None
        assert ops.rms_norm(torch.randn(1, 4), w, 1e-5) is None  # width mismatch


class TestRope:
    @pytest.mark.parametrize("tokens", [1, 5])
    def test_matches_rotate_half_on_strided_views(self, ops: FusedOps, tokens: int) -> None:
        torch.manual_seed(1)
        heads, kv_heads, dim = 6, 2, 16
        # Views of a projection output, as the attention layers build them (non-contiguous).
        q = torch.randn(1, tokens, heads, dim).transpose(1, 2)
        k = torch.randn(1, tokens, kv_heads, dim).transpose(1, 2)
        freqs = torch.randn(tokens, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        fused = ops.rope(q, k, cos, sin)
        assert fused is not None
        for got, x in zip(fused, (q, k), strict=True):
            expected = x * cos + rotate_half(x) * sin
            assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)

    def test_declines_batched_cos(self, ops: FusedOps) -> None:
        q = torch.randn(1, 2, 1, 8)
        assert ops.rope(q, q, torch.randn(2, 1, 8), torch.randn(2, 1, 8)) is None


class TestAttention:
    @pytest.mark.parametrize(
        ("n_q", "n_kv", "offset", "window"),
        [(1, 40, 39, 0), (1, 40, 39, 8), (3, 40, 37, 0), (4, 12, 8, 5), (1, 1, 0, 0)],
    )
    def test_matches_sdpa_with_a_causal_mask(
        self, ops: FusedOps, n_q: int, n_kv: int, offset: int, window: int
    ) -> None:
        torch.manual_seed(2)
        heads, kv_heads, dim = 8, 2, 32
        q = torch.randn(1, n_q, heads, dim).transpose(1, 2)
        cache_k = torch.randn(1, kv_heads, 64, dim)  # a cache with spare capacity, sliced
        cache_v = torch.randn(1, kv_heads, 64, dim)
        k, v = cache_k[:, :, :n_kv], cache_v[:, :, :n_kv]
        scale = 0.3
        q_idx, kv_idx = torch.arange(n_q).unsqueeze(1), torch.arange(n_kv).unsqueeze(0)
        mask = kv_idx <= q_idx + offset
        if window:
            mask &= kv_idx > q_idx + offset - window
        expected = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=True
        )
        fused = ops.attention(q, k, v, offset, scale, window)
        assert fused is not None
        assert torch.allclose(fused, expected, atol=1e-5, rtol=1e-5)

    def test_declines_prefills_batches_and_other_dtypes(self, ops: FusedOps) -> None:
        q, kv = torch.randn(1, 4, 20, 8), torch.randn(1, 2, 30, 8)
        assert ops.attention(q, kv, kv, 10, 0.3) is None  # a prefill: torch is faster
        assert ops.attention(q[:, :, :1], kv, kv, torch.tensor([3]), 0.3) is None  # batched
        assert ops.attention(q[:, :, :1].bfloat16(), kv, kv, 3, 0.3) is None


def _torch_probabilities(
    logits: torch.Tensor, generated: list[int], sampling: SamplingConfig
) -> torch.Tensor:
    """The existing torch pipeline's final distribution (what multinomial would draw from)."""
    x = RepetitionPenaltyFilter(sampling.repeat_penalty).apply(logits, generated)
    x = TopPFilter(sampling.top_p).apply(TopKFilter(sampling.top_k).apply(x / sampling.temperature))
    return torch.softmax(x, dim=-1)


class TestSampler:
    @pytest.mark.parametrize(
        "sampling",
        [
            SamplingConfig(temperature=0.7, top_k=10, top_p=0.9, repeat_penalty=1.3),
            SamplingConfig(temperature=1.0, top_k=5, top_p=1.0, repeat_penalty=1.0),
            SamplingConfig(temperature=0.5, top_k=0, top_p=1.0, repeat_penalty=1.1),
            SamplingConfig(temperature=1.3, top_k=40, top_p=0.5, repeat_penalty=1.0),
        ],
    )
    def test_draws_from_the_same_distribution_as_the_torch_pipeline(
        self, ops: FusedOps, sampling: SamplingConfig
    ) -> None:
        torch.manual_seed(3)
        logits = torch.randn(60) * 2
        generated = [3, 7, 7, 20]
        expected = _torch_probabilities(logits, generated, sampling)
        assert ops.can_sample(logits, sampling)
        draws = 6000
        counts = torch.zeros(60)
        for u in (torch.arange(draws) + 0.5) / draws:  # an even sweep of the unit interval
            counts[ops.sample(logits, generated, sampling, float(u))] += 1
        assert torch.allclose(counts / draws, expected, atol=2e-3)

    def test_ties_at_the_top_k_boundary_are_all_kept_like_the_torch_filter(
        self, ops: FusedOps
    ) -> None:
        logits = torch.tensor([5.0, 4.0, 4.0, 4.0, 1.0, 0.0])
        sampling = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0, repeat_penalty=1.0)
        expected = _torch_probabilities(logits, [], sampling)
        assert (expected > 0).sum() == 4  # 5.0 and all three 4.0s
        seen = {ops.sample(logits, [], sampling, (i + 0.5) / 400) for i in range(400)}
        assert seen == {0, 1, 2, 3}

    def test_top_p_without_top_k_stays_on_torch(self, ops: FusedOps) -> None:
        sampling = dataclasses.replace(SamplingConfig(), temperature=0.8, top_k=0, top_p=0.9)
        assert not ops.can_sample(torch.randn(10), sampling)
        assert not ops.can_sample(
            torch.randn(10), dataclasses.replace(sampling, temperature=0.0, top_k=5)
        )  # greedy is argmax elsewhere


def test_the_numba_backend_leaves_attention_to_torch_by_default() -> None:
    q, kv = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 30, 8)
    assert FusedOps(_NumbaBackend()).attention(q, kv, kv, 3, 0.3) is None
    assert FusedOps(_NumbaBackend(attention=True)).attention(q, kv, kv, 3, 0.3) is not None

"""Hybrid (recurrent) models on a GPU, checked without one: the recurrence falls back to the torch
reference loop on non-CPU tensors, caches follow their device, and a full forward through
placement (packed mode on the CPU device) equals the plain model."""

from pathlib import Path

import pytest
import torch

from app.architectures.nemotron_h import NemotronHArchitecture
from app.architectures.qwen35 import Qwen35Architecture
from app.architectures.qwen35_delta_kernels import (
    gated_delta_rule,
    gated_delta_rule_chunked,
    gated_delta_rule_reference,
)
from app.gguf.loader import GGUFModelLoader
from app.runtime.batch_cache import BatchedHybridCache
from app.runtime.mamba_cache import NemotronHHybridCache
from tests.tiny_gguf_nemotron_h_q8 import build_tiny_nemotron_h_q8_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf

META = torch.device("meta")


def _cache(device: torch.device) -> NemotronHHybridCache:
    return NemotronHHybridCache(["mamba", "attention"], (2, 4), (3, 8), (2, 4, 4), 8, device=device)


class TestHybridCacheDevice:
    def test_every_part_is_allocated_on_the_device_and_survives_a_fork(self) -> None:
        cache = _cache(META)
        assert cache.device == META and cache.conv_state[0].device == META
        forked = cache.fork_from(cache.snapshot())
        assert forked.device == META and forked.ssm_state[0].device == META
        assert BatchedHybridCache([cache]).length.device == META


def test_gated_delta_rule_off_the_cpu_uses_the_torch_loop() -> None:
    t, h, dk, dv = 3, 2, 4, 4
    out, state = gated_delta_rule(
        torch.empty(t, h, dk, device=META),
        torch.empty(t, h, dk, device=META),
        torch.empty(t, h, dv, device=META),
        torch.empty(t, h, device=META),
        torch.empty(t, h, device=META),
        torch.empty(h, dk, dv, device=META),
    )
    assert out.shape == (t, h, dv) and state.shape == (h, dk, dv) and out.device == META


def test_the_cpu_kernel_still_matches_the_reference() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(5, 2, 4) for _ in range(3))
    g, beta = -torch.rand(5, 2), torch.rand(5, 2)
    state = torch.randn(2, 4, 4)
    expected = gated_delta_rule_reference(q, k, v, g, beta, state.clone())
    actual = gated_delta_rule(q, k, v, g, beta, state.clone())
    assert torch.allclose(actual[0], expected[0], atol=1e-5)
    assert torch.allclose(actual[1], expected[1], atol=1e-5)


@pytest.mark.parametrize(("n_tokens", "chunk"), [(1, 16), (5, 16), (16, 16), (17, 16), (130, 64)])
def test_chunked_scan_matches_the_per_token_loop(n_tokens: int, chunk: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(n_tokens, 3, 8) * 0.3
    k = torch.nn.functional.normalize(torch.randn(n_tokens, 3, 8), dim=-1)
    v = torch.randn(n_tokens, 3, 8)
    g, beta, state = -torch.rand(n_tokens, 3) * 0.5, torch.rand(n_tokens, 3), torch.randn(3, 8, 8)
    expected = gated_delta_rule_reference(q, k, v, g, beta, state.clone())
    actual = gated_delta_rule_chunked(q, k, v, g, beta, state.clone(), chunk=chunk)
    assert torch.allclose(actual[0], expected[0], atol=1e-5)
    assert torch.allclose(actual[1], expected[1], atol=1e-5)


@pytest.mark.parametrize("name", ["qwen35", "nemotron_h"])
def test_packed_placement_forward_equals_the_plain_model(name: str, tmp_path: Path) -> None:
    if name == "qwen35":
        path, cls = build_tiny_qwen35_gguf(tmp_path / "m.gguf"), Qwen35Architecture
    else:
        path, cls = build_tiny_nemotron_h_q8_gguf(tmp_path / "m.gguf"), NemotronHArchitecture

    def load(packed: bool):
        model = cls.from_gguf(
            GGUFModelLoader(path, dtype=torch.float32),
            dtype=torch.float32,
            enable_quantized_native=packed,
        )
        model.place_on(torch.device("cpu"), packed_weights=packed)
        return model.eval()

    ids = torch.tensor([[1, 5, 9, 3]])
    plain, packed = load(False), load(True)
    with torch.no_grad():
        expected = plain(ids, kv_cache=plain.build_cache(8, torch.float32))
        actual = packed(ids, kv_cache=packed.build_cache(8, torch.float32))
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)

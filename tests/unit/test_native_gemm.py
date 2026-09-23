"""Correctness for the native C quantized matmul (app/native/, see ROADMAP.md's "In-house native (C)
quantized kernels" entry). Every kernel is cross-checked against the already oracle-validated
`*Strategy.dequantize()` + float matmul path, same approach as tests/unit/test_quantized_gemv.py.

Two checks per type:
- **exact**: activations that are already integers with a +-127 peak in every 256-value block
  survive the kernel's int8 activation quantization unchanged, so the result must match to float
  precision - any bit-layout mistake shows up here, not hidden inside rounding noise;
- **realistic**: random float activations, where int8 rounding (which llama.cpp has too) is allowed
  a small relative error.

The library is compiled into a per-session temp dir, so the tests never touch app/native/build/.
"""

import random
import shutil
import struct

import pytest
import torch

from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.constants import GGMLQuantizationType as T
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.native.gemm import NativeGemm
from app.native.library import NativeKernelLibrary

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler")


def _rand(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def _f16(rng: random.Random, lo: float, hi: float) -> bytes:
    return struct.pack("<e", rng.uniform(lo, hi))


# One random, structurally valid block per type (field order = ggml's struct / *Strategy).
_BLOCKS = {
    T.Q3_K: lambda r: _rand(r, 32 + 64 + 12) + _f16(r, 0.01, 1.0),
    T.Q4_K: lambda r: _f16(r, 0.01, 1.0) + _f16(r, 0.01, 1.0) + _rand(r, 12 + 128),
    T.Q5_K: lambda r: _f16(r, 0.01, 1.0) + _f16(r, 0.01, 1.0) + _rand(r, 12 + 32 + 128),
    T.Q6_K: lambda r: (
        _rand(r, 128 + 64)
        + struct.pack("<16b", *(r.randrange(-30, 31) for _ in range(16)))
        + _f16(r, 0.01, 1.0)
    ),
    T.Q8_0: lambda r: _f16(r, 0.01, 1.0) + _rand(r, 32),
}
_BLOCKS_PER_256 = {T.Q8_0: 8}


@pytest.fixture(scope="module")
def gemm(tmp_path_factory: pytest.TempPathFactory) -> NativeGemm:
    lib = NativeKernelLibrary(build_dir=tmp_path_factory.mktemp("native_build")).load()
    return NativeGemm(lib, n_threads=2)


def _weight(ggml_type: T, out_features: int, in_features: int, seed: int) -> bytes:
    rng = random.Random(seed)
    n_blocks = out_features * (in_features // 256) * _BLOCKS_PER_256.get(ggml_type, 1)
    return b"".join(_BLOCKS[ggml_type](rng) for _ in range(n_blocks))


def _reference(ggml_type: T, raw: bytes, x: torch.Tensor, out_f: int, in_f: int) -> torch.Tensor:
    strategy = QuantStrategyRegistry().get(ggml_type)
    weight = strategy.dequantize(memoryview(raw), out_f * in_f).reshape(out_f, in_f)
    return x.to(torch.float64) @ weight.to(torch.float64).T


def _integer_activations(n_tokens: int, in_features: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-127, 128, (n_tokens, in_features), generator=g).to(torch.float32)
    x[:, ::256] = 127.0  # pin every block's peak so the int8 scale is exactly 1
    return x


@pytest.mark.parametrize("ggml_type", list(_BLOCKS))
@pytest.mark.parametrize("n_tokens", [1, 3])
def test_exact_on_int8_representable_activations(
    gemm: NativeGemm, ggml_type: T, n_tokens: int
) -> None:
    out_f, in_f = 7, 512
    raw = _weight(ggml_type, out_f, in_f, seed=int(ggml_type))
    x = _integer_activations(n_tokens, in_f, seed=n_tokens)

    result = gemm.matmul(ggml_type, memoryview(raw), x, out_f, in_f)

    expected = _reference(ggml_type, raw, x, out_f, in_f)
    assert result.shape == (n_tokens, out_f)
    assert torch.allclose(result.to(torch.float64), expected, rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize("ggml_type", list(_BLOCKS))
def test_close_on_float_activations(gemm: NativeGemm, ggml_type: T) -> None:
    out_f, in_f = 16, 1024
    raw = _weight(ggml_type, out_f, in_f, seed=100 + int(ggml_type))
    x = torch.randn(2, in_f, generator=torch.Generator().manual_seed(7))

    result = gemm.matmul(ggml_type, memoryview(raw), x, out_f, in_f).to(torch.float64)

    expected = _reference(ggml_type, raw, x, out_f, in_f)
    rel_err = torch.linalg.norm(result - expected) / torch.linalg.norm(expected)
    assert rel_err < 1e-2


def test_supports_only_implemented_types_and_block_multiples(gemm: NativeGemm) -> None:
    assert gemm.supports(T.Q4_K, 3072)
    assert not gemm.supports(T.Q4_K, 3000)  # not a multiple of 256
    assert not gemm.supports(T.Q2_K, 3072)  # no native kernel yet - numba keeps it


def test_configure_numba_leaves_native_inactive() -> None:
    NativeGemm.configure("numba")
    assert NativeGemm.active() is None


@pytest.mark.parametrize("seq_len", [1, 4])
def test_quantized_linear_native_matches_numba_path(
    gemm: NativeGemm, monkeypatch: pytest.MonkeyPatch, seq_len: int
) -> None:
    """Same QuantizedLinear, same weights: the native backend must agree with today's Numba
    decode / dequant-prefill path it replaces."""
    out_f, in_f = 12, 512
    raw = memoryview(_weight(T.Q4_K, out_f, in_f, seed=42))
    x = torch.randn(1, seq_len, in_f, generator=torch.Generator().manual_seed(3))

    monkeypatch.setattr(NativeGemm, "_active", None)
    numba_layer = QuantizedLinear(out_f, in_f, T.Q4_K, raw, dtype=torch.float32)
    monkeypatch.setattr(NativeGemm, "_active", gemm)
    native_layer = QuantizedLinear(out_f, in_f, T.Q4_K, raw, dtype=torch.float32)

    expected = numba_layer(x)
    result = native_layer(x)

    assert result.shape == expected.shape == (1, seq_len, out_f)
    assert torch.linalg.norm(result - expected) / torch.linalg.norm(expected) < 1e-2


@pytest.mark.parametrize("ggml_type", list(_BLOCKS))
def test_decode_kernel_is_bit_identical_to_the_prefill_path(gemm: NativeGemm, ggml_type: T) -> None:
    """One token takes the fused per-token kernel (with SSSE3), several tokens the unpack-once
    path - every token must come out bit-identical either way, or a reply would depend on whether
    a token happened to be computed in prefill or decode."""
    out_f, in_f = 9, 768
    raw = memoryview(_weight(ggml_type, out_f, in_f, seed=77))
    x = torch.randn(6, in_f, generator=torch.Generator().manual_seed(5))

    prefill = gemm.matmul(ggml_type, raw, x, out_f, in_f)

    for t in range(6):
        assert torch.equal(gemm.matmul(ggml_type, raw, x[t : t + 1], out_f, in_f)[0], prefill[t])

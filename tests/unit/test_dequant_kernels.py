import random
import struct

import numpy as np
import pytest
import torch

from app.gguf.constants import GGMLQuantizationType
from app.gguf.dequant.legacy import (
    Q4_0Strategy,
    Q4_1Strategy,
    Q5_0Strategy,
    Q5_1Strategy,
    Q8_0Strategy,
)
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.dequant.simple import BF16Strategy, F16Strategy, F32Strategy
from app.server.errors import UnsupportedQuantTypeError


def _pack_nibbles(low: list[int], high: list[int]) -> bytes:
    return bytes((low[j] & 0x0F) | ((high[j] & 0x0F) << 4) for j in range(len(low)))


class TestF32Strategy:
    def test_dequantize_is_identity(self) -> None:
        values = [1.5, -2.5, 3.25, 0.0]
        raw = struct.pack("<4f", *values)

        result = F32Strategy().dequantize(memoryview(raw), n_elements=4)

        assert torch.allclose(result, torch.tensor(values))


class TestF16Strategy:
    def test_dequantize_upcasts_to_float32(self) -> None:
        values = np.array([1.5, -2.0, 0.5, 4.0], dtype="<f2")
        raw = values.tobytes()

        result = F16Strategy().dequantize(memoryview(raw), n_elements=4)

        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor([1.5, -2.0, 0.5, 4.0]))


class TestQ8_0Strategy:
    def test_dequantize_one_block(self) -> None:
        strategy = Q8_0Strategy()
        d = 0.5
        qs = list(range(-16, 16))
        raw = struct.pack("<e", d) + struct.pack("<32b", *qs)

        result = strategy.dequantize(memoryview(raw), n_elements=32)

        expected = torch.tensor([q * d for q in qs])
        assert torch.allclose(result, expected)

    def test_byte_length(self) -> None:
        assert Q8_0Strategy().byte_length(64) == 2 * 34


class TestQ4_0Strategy:
    def test_dequantize_one_block(self) -> None:
        strategy = Q4_0Strategy()
        d = 0.5
        low = list(range(16))
        high = list(range(15, -1, -1))
        raw = struct.pack("<e", d) + _pack_nibbles(low, high)

        result = strategy.dequantize(memoryview(raw), n_elements=32)

        expected = torch.tensor([(n - 8) * d for n in low] + [(n - 8) * d for n in high])
        assert torch.allclose(result, expected)


class TestQ4_1Strategy:
    def test_dequantize_one_block(self) -> None:
        strategy = Q4_1Strategy()
        d, m = 0.5, 1.0
        low = list(range(16))
        high = list(range(15, -1, -1))
        raw = struct.pack("<e", d) + struct.pack("<e", m) + _pack_nibbles(low, high)

        result = strategy.dequantize(memoryview(raw), n_elements=32)

        expected = torch.tensor([n * d + m for n in low] + [n * d + m for n in high])
        assert torch.allclose(result, expected)


class TestBF16Strategy:
    def test_dequantize_widens_top_16_bits(self) -> None:
        values = np.array([1.5, -2.0, 0.5, 4.0], dtype="<f4")
        raw = (values.view("<u4") >> 16).astype("<u2").tobytes()

        result = BF16Strategy().dequantize(memoryview(raw), n_elements=4)

        assert result.dtype == torch.float32
        assert torch.allclose(result, torch.tensor(values), atol=1e-2)


def _ref_dequant_q5_0(d: float, qh: int, qs: list[int]) -> list[float]:
    lows, highs = [], []
    for j in range(16):
        xh_0 = ((qh >> (j + 0)) << 4) & 0x10
        xh_1 = (qh >> (j + 12)) & 0x10
        lows.append((((qs[j] & 0x0F) | xh_0) - 16) * d)
        highs.append((((qs[j] >> 4) | xh_1) - 16) * d)
    return lows + highs


def _ref_dequant_q5_1(d: float, m: float, qh: int, qs: list[int]) -> list[float]:
    lows, highs = [], []
    for j in range(16):
        xh_0 = ((qh >> (j + 0)) << 4) & 0x10
        xh_1 = (qh >> (j + 12)) & 0x10
        lows.append(((qs[j] & 0x0F) | xh_0) * d + m)
        highs.append(((qs[j] >> 4) | xh_1) * d + m)
    return lows + highs


class TestQ5_0Strategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(45)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = struct.unpack("<e", struct.pack("<e", rng.uniform(0.01, 2.0)))[0]
            qh = rng.randrange(0, 2**32)
            qs = [rng.randrange(0, 256) for _ in range(16)]
            raw += struct.pack("<e", d) + struct.pack("<I", qh) + bytes(qs)
            expected.extend(_ref_dequant_q5_0(d, qh, qs))

        result = Q5_0Strategy().dequantize(memoryview(raw), n_elements=32 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestQ5_1Strategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(46)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = struct.unpack("<e", struct.pack("<e", rng.uniform(0.01, 2.0)))[0]
            m = struct.unpack("<e", struct.pack("<e", rng.uniform(0.01, 1.0)))[0]
            qh = rng.randrange(0, 2**32)
            qs = [rng.randrange(0, 256) for _ in range(16)]
            raw += struct.pack("<e", d) + struct.pack("<e", m) + struct.pack("<I", qh) + bytes(qs)
            expected.extend(_ref_dequant_q5_1(d, m, qh, qs))

        result = Q5_1Strategy().dequantize(memoryview(raw), n_elements=32 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestQuantStrategyRegistry:
    def test_resolves_known_types(self) -> None:
        registry = QuantStrategyRegistry()

        assert isinstance(registry.get(GGMLQuantizationType.F32), F32Strategy)
        assert isinstance(registry.get(GGMLQuantizationType.Q8_0), Q8_0Strategy)

    def test_resolves_m10_added_types(self) -> None:
        registry = QuantStrategyRegistry()

        for quant_type in (
            GGMLQuantizationType.BF16,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q2_K,
            GGMLQuantizationType.Q3_K,
            GGMLQuantizationType.Q8_K,
        ):
            assert registry.get(quant_type) is not None

    def test_unsupported_known_type_raises(self) -> None:
        with pytest.raises(UnsupportedQuantTypeError):
            QuantStrategyRegistry().get(GGMLQuantizationType.IQ2_XXS)

    def test_unrecognized_type_id_raises(self) -> None:
        with pytest.raises(UnsupportedQuantTypeError):
            QuantStrategyRegistry().get(9999)

    def test_supported_names_lists_every_registered_quant_type(self) -> None:
        names = QuantStrategyRegistry().supported_names()

        assert "F32" in names
        assert "Q4_K" in names
        assert "IQ2_XXS" not in names  # a real but genuinely unsupported type
        assert len(names) == len(QuantStrategyRegistry._STRATEGIES)

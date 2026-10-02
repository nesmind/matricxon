"""Pure-torch dequantization of packed GGUF blocks, so it runs on whatever device holds the bytes
(the GPU, for `Settings.gpu_weight_mode="packed"`). Each function mirrors the numpy `QuantStrategy`
of the same type bit for bit - `tests/unit/test_torch_dequant.py` checks them against it - and takes
`blocks`: a `(n_blocks, type_size)` uint8 tensor, returning `(n_blocks * block_size,)` float32.
"""

from collections.abc import Callable

import torch

from app.gguf.constants import GGMLQuantizationType as T
from app.gguf.dequant.iq_ternary import _KVALUES_IQ4NL
from app.gguf.dequant.registry import QuantStrategyRegistry


def _f16(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """Little-endian f16 at byte `start` of every block -> float32 `(n_blocks, 1)`."""
    return blocks[:, start : start + 2].contiguous().view(torch.float16).float()


def _nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Legacy 32-element blocks: low nibbles are elements 0-15, high nibbles 16-31."""
    return torch.cat([packed & 0x0F, packed >> 4], dim=1).float()


def _qh32(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """The 4-byte little-endian high-bit mask of a Q5_0/Q5_1 block, `(n_blocks, 1)` int64."""
    b = blocks[:, start : start + 4].long()
    return (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)).unsqueeze(1)


def _fifth_bit(qh: torch.Tensor) -> torch.Tensor:
    """Q5 high bits laid out like `_nibbles`: bit j for element j, bit j+12 (>>4 style) for 16+j."""
    j = torch.arange(16, device=qh.device)
    low = ((qh >> j) << 4) & 0x10
    high = (qh >> (j + 12)) & 0x10
    return torch.cat([low, high], dim=1).float()


def q8_0(blocks: torch.Tensor) -> torch.Tensor:
    return (blocks[:, 2:].contiguous().view(torch.int8).float() * _f16(blocks, 0)).reshape(-1)


def q4_0(blocks: torch.Tensor) -> torch.Tensor:
    return ((_nibbles(blocks[:, 2:]) - 8.0) * _f16(blocks, 0)).reshape(-1)


def q4_1(blocks: torch.Tensor) -> torch.Tensor:
    return (_nibbles(blocks[:, 4:]) * _f16(blocks, 0) + _f16(blocks, 2)).reshape(-1)


def q5_0(blocks: torch.Tensor) -> torch.Tensor:
    quant = _nibbles(blocks[:, 6:]) + _fifth_bit(_qh32(blocks, 2))
    return ((quant - 16.0) * _f16(blocks, 0)).reshape(-1)


def q5_1(blocks: torch.Tensor) -> torch.Tensor:
    quant = _nibbles(blocks[:, 8:]) + _fifth_bit(_qh32(blocks, 4))
    return (quant * _f16(blocks, 0) + _f16(blocks, 2)).reshape(-1)


def _scale_min_k4(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ggml's get_scale_min_k4 on `(n_blocks, 12)` -> (sc, m), each `(n_blocks, 8)` float32."""
    s = scales.long()
    sc = torch.cat([s[:, 0:4] & 0x3F, (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)], dim=1)
    m = torch.cat([s[:, 4:8] & 0x3F, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], dim=1)
    return sc.float(), m.float()


def q4_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    d, dmin = _f16(blocks, 0).view(n, 1, 1, 1), _f16(blocks, 2).view(n, 1, 1, 1)
    sc, m = _scale_min_k4(blocks[:, 4:16])
    qs = blocks[:, 16:].reshape(n, 4, 32)
    quants = torch.stack([qs & 0x0F, qs >> 4], dim=2).float()  # (n, 4, 2, 32)
    return (d * sc.view(n, 4, 2, 1) * quants - dmin * m.view(n, 4, 2, 1)).reshape(-1)


def q5_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    d, dmin = _f16(blocks, 0).view(n, 1, 1, 1), _f16(blocks, 2).view(n, 1, 1, 1)
    sc, m = _scale_min_k4(blocks[:, 4:16])
    qh = blocks[:, 16:48].long().unsqueeze(1)  # (n, 1, 32)
    ql = blocks[:, 48:].reshape(n, 4, 32).long()
    shifts = (torch.arange(4, device=blocks.device) * 2).view(1, 4, 1)
    low = (ql & 0x0F) + (((qh >> shifts) & 1) << 4)
    high = (ql >> 4) + (((qh >> (shifts + 1)) & 1) << 4)
    quants = torch.stack([low, high], dim=2).float()  # (n, 4, 2, 32)
    return (d * sc.view(n, 4, 2, 1) * quants - dmin * m.view(n, 4, 2, 1)).reshape(-1)


def q6_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    ql = blocks[:, :128].reshape(n, 2, 64).long()
    qh = blocks[:, 128:192].reshape(n, 2, 32).long()
    scales = blocks[:, 192:208].contiguous().view(torch.int8).reshape(n, 2, 4, 2).float()
    d = _f16(blocks, 208).view(n, 1, 1, 1)
    lo, hi = ql[:, :, :32], ql[:, :, 32:]
    quants = (
        torch.stack(
            [
                (lo & 0x0F) | (((qh >> 0) & 3) << 4),
                (hi & 0x0F) | (((qh >> 2) & 3) << 4),
                (lo >> 4) | (((qh >> 4) & 3) << 4),
                (hi >> 4) | (((qh >> 6) & 3) << 4),
            ],
            dim=2,
        ).float()
        - 32.0
    )  # (n, 2, 4, 32)
    scale = scales.repeat_interleave(16, dim=-1)  # (n, 2, 4, 32): 16 elements per int8 scale
    return (d * scale * quants).reshape(-1)


def q8_k(blocks: torch.Tensor) -> torch.Tensor:
    d = blocks[:, 0:4].contiguous().view(torch.float32)
    return (blocks[:, 4:260].contiguous().view(torch.int8).float() * d).reshape(-1)


_SHIFTS = (0, 2, 4, 6)


def _two_bit(qs: torch.Tensor) -> torch.Tensor:
    """Q2_K/Q3_K: `(n, 64)` bytes -> 2-bit values `(n, 2, 4, 32)`: (half, shift step, element)."""
    shifts = torch.tensor(_SHIFTS, device=qs.device).view(1, 1, 4, 1)
    return (qs.reshape(-1, 2, 1, 32).long() >> shifts) & 3


def _per16(scale: torch.Tensor) -> torch.Tensor:
    """`(n, 16)` one value per 16 elements -> `(n, 2, 4, 32)` matching `_two_bit`'s layout."""
    return scale.reshape(-1, 2, 4, 2).repeat_interleave(16, dim=-1)


def q2_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    scales = blocks[:, 0:16].long()
    d, dmin = _f16(blocks, 80).view(n, 1, 1, 1), _f16(blocks, 82).view(n, 1, 1, 1)
    quants = _two_bit(blocks[:, 16:80]).float()
    sc, mn = _per16((scales & 0x0F).float()), _per16((scales >> 4).float())
    return (d * sc * quants - dmin * mn).reshape(-1)


def _q3k_scales(raw: torch.Tensor) -> torch.Tensor:
    """ggml's 12-byte packed signed 6-bit scales -> `(n, 16)` float, already minus 32. Works on
    32-bit words (bits carry across bytes before the masks), as the numpy version does."""
    b = raw.long().reshape(-1, 3, 4)
    w = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)  # (n, 3) u32
    k1, k2 = 0x03030303, 0x0F0F0F0F
    w0, w1, tmp = w[:, 0], w[:, 1], w[:, 2]
    words = torch.stack(
        [
            (w0 & k2) | ((tmp & k1) << 4),
            (w1 & k2) | (((tmp >> 2) & k1) << 4),
            ((w0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
            ((w1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
        ],
        dim=1,
    )
    shifts = torch.tensor([0, 8, 16, 24], device=raw.device).view(1, 1, 4)
    as_bytes = ((words.unsqueeze(-1) >> shifts) & 0xFF).reshape(-1, 16)
    return (torch.where(as_bytes > 127, as_bytes - 256, as_bytes) - 32).float()


def q3_k(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    hmask = blocks[:, 0:32].long().reshape(n, 1, 1, 32)
    d = _f16(blocks, 108).view(n, 1, 1, 1)
    bits = _two_bit(blocks[:, 32:96])
    # hmask bit (4 * half + step) belongs to quant group (half, step).
    mask = (
        1
        << (
            torch.arange(2, device=blocks.device).view(2, 1) * 4
            + torch.arange(4, device=blocks.device)
        )
    ).view(1, 2, 4, 1)
    sign = torch.where((hmask & mask) != 0, 0, 4)
    scale = _per16(_q3k_scales(blocks[:, 96:108]))
    return (d * scale * (bits - sign).float()).reshape(-1)


def _iq4_table(device: torch.device) -> torch.Tensor:
    return torch.tensor(_KVALUES_IQ4NL.tolist(), dtype=torch.float32, device=device)


def iq4_nl(blocks: torch.Tensor) -> torch.Tensor:
    table, qs = _iq4_table(blocks.device), blocks[:, 2:18].long()
    values = torch.cat([table[qs & 0x0F], table[qs >> 4]], dim=1)
    return (values * _f16(blocks, 0)).reshape(-1)


def iq4_xs(blocks: torch.Tensor) -> torch.Tensor:
    n = blocks.shape[0]
    ib = torch.arange(8, device=blocks.device)
    scales_h = blocks[:, 2].long() | (blocks[:, 3].long() << 8)
    scales_l = blocks[:, 4:8].long()
    low = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0x0F
    high = ((scales_h.unsqueeze(1) >> (2 * ib)) & 3) << 4
    dl = (_f16(blocks, 0) * ((low | high) - 32).float()).view(n, 8, 1)
    table, qs = _iq4_table(blocks.device), blocks[:, 8:136].reshape(n, 8, 16).long()
    values = torch.cat([table[qs & 0x0F], table[qs >> 4]], dim=2)  # (n, 8, 32)
    return (values * dl).reshape(-1)


_KERNELS: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {
    T.Q8_0: q8_0,
    T.Q4_0: q4_0,
    T.Q4_1: q4_1,
    T.Q5_0: q5_0,
    T.Q5_1: q5_1,
    T.Q4_K: q4_k,
    T.Q5_K: q5_k,
    T.Q6_K: q6_k,
    T.Q2_K: q2_k,
    T.Q3_K: q3_k,
    T.Q8_K: q8_k,
    T.IQ4_NL: iq4_nl,
    T.IQ4_XS: iq4_xs,
}


class TorchDequantizer:
    """Which quant types dequantize on-device, and their block geometry."""

    @staticmethod
    def supports(ggml_type: int) -> bool:
        return ggml_type in _KERNELS

    @staticmethod
    def geometry(ggml_type: int) -> tuple[int, int]:
        """`(block_size, type_size)` of `ggml_type`."""
        strategy = QuantStrategyRegistry().get(ggml_type)
        return strategy.block_size, strategy.type_size

    @staticmethod
    def dequantize(blocks: torch.Tensor, ggml_type: int) -> torch.Tensor:
        return _KERNELS[ggml_type](blocks)

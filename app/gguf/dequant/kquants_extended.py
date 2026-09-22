import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy

_QK_K = 256
_K_SCALE_SIZE = 12


def _unpack_q3k_scales(scales_raw: np.ndarray) -> np.ndarray:
    """scales_raw: (n_blocks, 12) uint8 -> (n_blocks, 16) int32, already -32.

    Ports ggml's block_q3_K aux[4]/kmask1/kmask2 trick: 16 signed 6-bit scales
    packed into 12 bytes as 3 little-endian uint32 words, reassembled into 4
    words, then reread byte-by-byte as int8. Must operate at uint32 granularity
    (not per-byte) since the reference shifts the full 32-bit word, letting bits
    carry across byte boundaries before the mask zeroes them back out.
    """
    words = scales_raw.copy().view("<u4")  # (n_blocks, 3)
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0F0F0F0F)
    w0, w1, tmp = words[:, 0], words[:, 1], words[:, 2]
    w2 = ((w0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    w3 = ((w1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    w0n = (w0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    w1n = (w1 & kmask2) | (((tmp >> 2) & kmask1) << 4)

    packed = np.stack([w0n, w1n, w2, w3], axis=1).astype("<u4")  # (n_blocks, 4)
    return packed.view(np.int8).reshape(scales_raw.shape[0], 16).astype(np.int32) - 32


class Q2_KStrategy(QuantStrategy):
    """256-element super-block: {scales[16]; qs[64] (2-bit); d, dmin: f16}.

    16 sub-blocks of 16 elements, each scales byte packing a 4-bit scale
    (low nibble) and 4-bit min (high nibble): value = d*sc*quant - dmin*min,
    no sign offset on the raw 2-bit quant itself (unlike Q3_K/Q6_K).
    """

    block_size = _QK_K
    type_size = _QK_K // 16 + _QK_K // 4 + 2 + 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        offset = 0
        scales_raw = blocks[:, offset : offset + _QK_K // 16]
        offset += _QK_K // 16
        qs = blocks[:, offset : offset + _QK_K // 4]
        offset += _QK_K // 4
        d = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        offset += 2
        dmin = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)

        sc = (scales_raw & 0x0F).astype(np.float32)  # (n_blocks, 16)
        mn = (scales_raw >> 4).astype(np.float32)  # (n_blocks, 16)

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        is_ = 0
        for n in range(2):
            q_block = qs[:, n * 32 : (n + 1) * 32]  # (n_blocks, 32)
            base = n * 128
            shift = 0
            for _j in range(4):
                for half, lo, hi in ((0, 0, 16), (16, 16, 32)):
                    dl = d[:, 0] * sc[:, is_]
                    ml = dmin[:, 0] * mn[:, is_]
                    bits = ((q_block[:, lo:hi] >> shift) & 3).astype(np.float32)
                    out[:, base + half : base + half + 16] = dl.reshape(-1, 1) * bits - ml.reshape(
                        -1, 1
                    )
                    is_ += 1
                base += 32
                shift += 2
        return torch.from_numpy(out.reshape(-1))


class Q3_KStrategy(QuantStrategy):
    """256-element super-block: {hmask[32]; qs[64]; scales[12] (6-bit signed); d: f16}.

    Each 3-bit quant is 2 low bits from qs plus a high bit from hmask (bit set
    -> +0, bit clear -> -4, ggml's inverted convention); scale = sc-32, with sc
    from the same 12-byte packing as Q2_K/Q4_K/Q5_K but reassembled at uint32
    granularity (see `_unpack_q3k_scales`).
    """

    block_size = _QK_K
    type_size = _QK_K // 8 + _QK_K // 4 + _K_SCALE_SIZE + 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        offset = 0
        hmask = blocks[:, offset : offset + _QK_K // 8]
        offset += _QK_K // 8
        qs = blocks[:, offset : offset + _QK_K // 4]
        offset += _QK_K // 4
        scales_raw = blocks[:, offset : offset + _K_SCALE_SIZE]
        offset += _K_SCALE_SIZE
        d_all = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)

        scales = _unpack_q3k_scales(scales_raw)  # (n_blocks, 16)

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        is_ = 0
        m = 1
        for n in range(2):
            q_block = qs[:, n * 32 : (n + 1) * 32]  # (n_blocks, 32)
            base = n * 128
            shift = 0
            for _j in range(4):
                for half, lo, hi in ((0, 0, 16), (16, 16, 32)):
                    dl = d_all[:, 0] * scales[:, is_]
                    bits = (q_block[:, lo:hi] >> shift).astype(np.int32) & 3
                    sign = np.where((hmask[:, lo:hi] & m) != 0, 0, 4)
                    out[:, base + half : base + half + 16] = dl.reshape(-1, 1) * (bits - sign)
                    is_ += 1
                base += 32
                shift += 2
                m <<= 1
        return torch.from_numpy(out.reshape(-1))


class Q8_KStrategy(QuantStrategy):
    """256-element super-block: {d: f32; qs[256] (int8); bsums[16] (int16)}.

    Plain per-block scale, no sub-block scales/mins - value[i] = qs[i] * d.
    `bsums` (precomputed 16-lane sums, used by ggml's fused dot-product
    kernels for K-quant matmuls) doesn't affect the dequantized value.
    """

    block_size = _QK_K
    type_size = 4 + _QK_K + (_QK_K // 16) * 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        d = blocks[:, 0:4].copy().view("<f4")
        qs = blocks[:, 4 : 4 + _QK_K].view(np.int8).astype(np.float32)

        values = qs * d
        return torch.from_numpy(values.reshape(-1))

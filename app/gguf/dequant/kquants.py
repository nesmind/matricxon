import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy

_QK_K = 256
_K_SCALE_SIZE = 12


def _get_scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scales: (n_blocks, 12) uint8 -> (sc, m), each (n_blocks, 8) uint8.

    Ports ggml's get_scale_min_k4: 8 sub-block (6-bit scale, 6-bit min) pairs
    packed into 12 bytes, used by both Q4_K and Q5_K.
    """
    n_blocks = scales.shape[0]
    sc = np.empty((n_blocks, 8), dtype=np.uint8)
    m = np.empty((n_blocks, 8), dtype=np.uint8)
    for j in range(4):
        sc[:, j] = scales[:, j] & 0x3F
        m[:, j] = scales[:, j + 4] & 0x3F
    for j in range(4, 8):
        sc[:, j] = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, m


class Q4_KStrategy(QuantStrategy):
    """256-element super-block: {d, dmin: f16; scales[12]; qs[128] (4-bit)}.

    8 sub-blocks of 32 elements, each with its own 6-bit-packed scale/min
    (see _get_scale_min_k4): value = d*sc[sub]*nibble - dmin*m[sub].
    """

    block_size = _QK_K
    type_size = 2 + 2 + _K_SCALE_SIZE + _QK_K // 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        d = blocks[:, 0:2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        dmin = blocks[:, 2:4].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        scales = blocks[:, 4 : 4 + _K_SCALE_SIZE]
        qs = blocks[:, 4 + _K_SCALE_SIZE :].reshape(n_blocks, 4, 32)

        sc, m = _get_scale_min_k4(scales)
        sc = sc.astype(np.float32)
        m = m.astype(np.float32)

        low = (qs & 0x0F).astype(np.float32)
        high = (qs >> 4).astype(np.float32)

        sc_low = sc[:, 0::2].reshape(n_blocks, 4, 1)
        m_low = m[:, 0::2].reshape(n_blocks, 4, 1)
        sc_high = sc[:, 1::2].reshape(n_blocks, 4, 1)
        m_high = m[:, 1::2].reshape(n_blocks, 4, 1)

        d3 = d.reshape(n_blocks, 1, 1)
        dmin3 = dmin.reshape(n_blocks, 1, 1)

        value_low = d3 * sc_low * low - dmin3 * m_low
        value_high = d3 * sc_high * high - dmin3 * m_high

        combined = np.stack([value_low, value_high], axis=2)  # (n_blocks, 4, 2, 32)
        return torch.from_numpy(combined.reshape(-1))


class Q5_KStrategy(QuantStrategy):
    """256-element super-block: {d, dmin; scales[12]; qh[32]; qs[128]}.

    Like Q4_K but each 4-bit nibble gets a 5th bit from qh (adds 16 when set),
    and qh's 32 bytes are reused across all 4 chunks with different bit pairs.
    """

    block_size = _QK_K
    type_size = 2 + 2 + _K_SCALE_SIZE + _QK_K // 8 + _QK_K // 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        offset = 0
        d = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        offset += 2
        dmin = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        offset += 2
        scales = blocks[:, offset : offset + _K_SCALE_SIZE]
        offset += _K_SCALE_SIZE
        qh = blocks[:, offset : offset + _QK_K // 8]
        offset += _QK_K // 8
        ql = blocks[:, offset:].reshape(n_blocks, 4, 32)

        sc, m = _get_scale_min_k4(scales)
        sc = sc.astype(np.float32)
        m = m.astype(np.float32)

        out = np.empty((n_blocks, 4, 2, 32), dtype=np.float32)
        for c in range(4):
            bit_low = ((qh >> (2 * c)) & 1).astype(np.float32)
            bit_high = ((qh >> (2 * c + 1)) & 1).astype(np.float32)

            low_val = (ql[:, c, :] & 0x0F).astype(np.float32) + bit_low * 16.0
            high_val = (ql[:, c, :] >> 4).astype(np.float32) + bit_high * 16.0

            sc_low = sc[:, 2 * c].reshape(n_blocks, 1)
            m_low = m[:, 2 * c].reshape(n_blocks, 1)
            sc_high = sc[:, 2 * c + 1].reshape(n_blocks, 1)
            m_high = m[:, 2 * c + 1].reshape(n_blocks, 1)

            out[:, c, 0, :] = d * sc_low * low_val - dmin * m_low
            out[:, c, 1, :] = d * sc_high * high_val - dmin * m_high

        return torch.from_numpy(out.reshape(-1))


class Q6_KStrategy(QuantStrategy):
    """256-element super-block: {ql[128]; qh[64]; scales[16] (int8); d: f16}.

    16 sub-blocks of 16 elements, each with its own signed int8 scale;
    each 6-bit quant is 4 low bits from ql + 2 high bits from qh, minus 32.
    """

    block_size = _QK_K
    type_size = _QK_K // 2 + _QK_K // 4 + _QK_K // 16 + 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        offset = 0
        ql = blocks[:, offset : offset + _QK_K // 2].reshape(n_blocks, 2, 64)
        offset += _QK_K // 2
        qh = blocks[:, offset : offset + _QK_K // 4].reshape(n_blocks, 2, 32)
        offset += _QK_K // 4
        scales = blocks[:, offset : offset + _QK_K // 16].view(np.int8).reshape(n_blocks, 2, 8)
        offset += _QK_K // 16
        d = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)

        halves = []
        for h in range(2):
            ql_h, qh_h = ql[:, h, :], qh[:, h, :]
            ql_lo, ql_hi = ql_h[:, 0:32], ql_h[:, 32:64]
            sc_h = scales[:, h, :].astype(np.float32)

            q1 = ((ql_lo & 0x0F) | (((qh_h >> 0) & 3) << 4)).astype(np.float32) - 32.0
            q2 = ((ql_hi & 0x0F) | (((qh_h >> 2) & 3) << 4)).astype(np.float32) - 32.0
            q3 = ((ql_lo >> 4) | (((qh_h >> 4) & 3) << 4)).astype(np.float32) - 32.0
            q4 = ((ql_hi >> 4) | (((qh_h >> 6) & 3) << 4)).astype(np.float32) - 32.0

            values = []
            for q, lo_idx, hi_idx in ((q1, 0, 1), (q2, 2, 3), (q3, 4, 5), (q4, 6, 7)):
                scale = np.concatenate(
                    [
                        np.repeat(sc_h[:, lo_idx : lo_idx + 1], 16, axis=1),
                        np.repeat(sc_h[:, hi_idx : hi_idx + 1], 16, axis=1),
                    ],
                    axis=1,
                )
                values.append(d * scale * q)
            halves.append(np.concatenate(values, axis=1))  # (n_blocks, 128)

        combined = np.concatenate(halves, axis=1)  # (n_blocks, 256)
        return torch.from_numpy(combined.reshape(-1))

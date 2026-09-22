import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy

_QK8_0 = 32
_QK4_0 = 32
_QK4_1 = 32
_QK5_0 = 32
_QK5_1 = 32


class Q8_0Strategy(QuantStrategy):
    """Block layout: {ggml_half d; int8_t qs[32];} -> value[i] = qs[i] * d."""

    block_size = _QK8_0
    type_size = 2 + _QK8_0

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        scales = blocks[:, :2].copy().view("<f2").astype("<f4")
        quants = blocks[:, 2:].view(np.int8).astype(np.float32)

        values = quants * scales
        return torch.from_numpy(values.reshape(-1))


class Q4_0Strategy(QuantStrategy):
    """Block layout: {ggml_half d; uint8_t qs[16];} (2 nibbles/byte) ->

    value[i] = (nibble - 8) * d, for i in [0, 32).
    """

    block_size = _QK4_0
    type_size = 2 + _QK4_0 // 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        scales = blocks[:, :2].copy().view("<f2").astype("<f4")
        packed = blocks[:, 2:]

        low = (packed & 0x0F).astype(np.float32) - 8.0
        high = (packed >> 4).astype(np.float32) - 8.0
        nibbles = np.empty((n_blocks, self.block_size), dtype=np.float32)
        nibbles[:, : self.block_size // 2] = low
        nibbles[:, self.block_size // 2 :] = high

        values = nibbles * scales
        return torch.from_numpy(values.reshape(-1))


class Q4_1Strategy(QuantStrategy):
    """Block layout: {ggml_half d; ggml_half m; uint8_t qs[16];} ->

    value[i] = nibble * d + m, for i in [0, 32).
    """

    block_size = _QK4_1
    type_size = 2 + 2 + _QK4_1 // 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        scales = blocks[:, :2].copy().view("<f2").astype("<f4")
        mins = blocks[:, 2:4].copy().view("<f2").astype("<f4")
        packed = blocks[:, 4:]

        low = (packed & 0x0F).astype(np.float32)
        high = (packed >> 4).astype(np.float32)
        nibbles = np.empty((n_blocks, self.block_size), dtype=np.float32)
        nibbles[:, : self.block_size // 2] = low
        nibbles[:, self.block_size // 2 :] = high

        values = nibbles * scales + mins
        return torch.from_numpy(values.reshape(-1))


class Q5_0Strategy(QuantStrategy):
    """Block layout: {ggml_half d; uint8_t qh[4]; uint8_t qs[16];} ->

    5-bit quant = 4-bit nibble (qs) + a 5th bit from qh's 32-bit mask,
    value[i] = (quant - 16) * d, for i in [0, 32).
    """

    block_size = _QK5_0
    type_size = 2 + 4 + _QK5_0 // 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        scales = blocks[:, :2].copy().view("<f2").astype("<f4")
        qh = blocks[:, 2:6].copy().view("<u4").reshape(n_blocks).astype(np.uint32)
        packed = blocks[:, 6:]

        j = np.arange(self.block_size // 2)
        high_low = (((qh[:, None] >> j) << 4) & 0x10).astype(np.float32)
        high_high = ((qh[:, None] >> (j + 12)) & 0x10).astype(np.float32)

        low = (packed & 0x0F).astype(np.float32) + high_low - 16.0
        high = (packed >> 4).astype(np.float32) + high_high - 16.0
        nibbles = np.empty((n_blocks, self.block_size), dtype=np.float32)
        nibbles[:, : self.block_size // 2] = low
        nibbles[:, self.block_size // 2 :] = high

        values = nibbles * scales
        return torch.from_numpy(values.reshape(-1))


class Q5_1Strategy(QuantStrategy):
    """Block layout: {ggml_half d; ggml_half m; uint8_t qh[4]; uint8_t qs[16];} ->

    5-bit quant = 4-bit nibble (qs) + a 5th bit from qh's 32-bit mask,
    value[i] = quant * d + m, for i in [0, 32).
    """

    block_size = _QK5_1
    type_size = 2 + 2 + 4 + _QK5_1 // 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        scales = blocks[:, :2].copy().view("<f2").astype("<f4")
        mins = blocks[:, 2:4].copy().view("<f2").astype("<f4")
        qh = blocks[:, 4:8].copy().view("<u4").reshape(n_blocks).astype(np.uint32)
        packed = blocks[:, 8:]

        j = np.arange(self.block_size // 2)
        high_low = (((qh[:, None] >> j) << 4) & 0x10).astype(np.float32)
        high_high = ((qh[:, None] >> (j + 12)) & 0x10).astype(np.float32)

        low = (packed & 0x0F).astype(np.float32) + high_low
        high = (packed >> 4).astype(np.float32) + high_high
        nibbles = np.empty((n_blocks, self.block_size), dtype=np.float32)
        nibbles[:, : self.block_size // 2] = low
        nibbles[:, self.block_size // 2 :] = high

        values = nibbles * scales + mins
        return torch.from_numpy(values.reshape(-1))

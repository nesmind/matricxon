import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy


class F32Strategy(QuantStrategy):
    block_size = 1
    type_size = 4

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        array = np.frombuffer(raw, dtype="<f4", count=n_elements)
        return torch.from_numpy(array.copy())


class F16Strategy(QuantStrategy):
    block_size = 1
    type_size = 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        array = np.frombuffer(raw, dtype="<f2", count=n_elements)
        return torch.from_numpy(array.astype("<f4"))


class BF16Strategy(QuantStrategy):
    """bfloat16: the top 16 bits of an IEEE754 float32 (sign + 8-bit exponent

    + 7-bit mantissa). Widened back to float32 by left-shifting into the
    upper half of a uint32 and reinterpreting - exact, since bf16 only drops
    float32's low mantissa bits rather than re-encoding them.
    """

    block_size = 1
    type_size = 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        raw16 = np.frombuffer(raw, dtype="<u2", count=n_elements)
        widened = raw16.astype(np.uint32) << 16
        array = widened.view("<f4")
        return torch.from_numpy(array.copy())

from abc import ABC, abstractmethod

import torch


class QuantStrategy(ABC):
    """Converts a GGUF tensor's raw on-disk bytes into a dequantized tensor.

    Mirrors ggml's per-type block layout: `block_size` elements are packed
    into `type_size` bytes. Implementations operate on whole tensors (all
    blocks at once via vectorized tensor ops), never per-element Python loops
    - these tensors run to millions of elements.
    """

    block_size: int
    type_size: int

    def byte_length(self, n_elements: int) -> int:
        if n_elements % self.block_size != 0:
            raise ValueError(
                f"{type(self).__name__}: n_elements={n_elements} is not a multiple of "
                f"block_size={self.block_size}"
            )
        return (n_elements // self.block_size) * self.type_size

    @abstractmethod
    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        """Returns a flat 1-D float32 tensor of length n_elements."""

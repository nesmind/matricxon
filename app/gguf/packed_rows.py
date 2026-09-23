"""Row-level access to a packed (still quantized) GGUF weight. Every GGUF quantization type packs
each row of a 2-D weight into its own whole number of blocks, so row `i` is simply the byte range
`[i * row_bytes, (i + 1) * row_bytes)` - rows can be read or reordered without dequantizing
anything. That's what lets a token-embedding lookup dequantize only the rows it needs, and lets
`unpermute_rope_rows`'s pure row permutation run on the packed bytes of attn_q/attn_k.
"""

import numpy as np
import torch


class PackedRows:
    def __init__(self, raw: memoryview, n_rows: int) -> None:
        if n_rows <= 0 or len(raw) % n_rows:
            raise ValueError(f"{len(raw)} packed bytes don't split into {n_rows} equal rows")
        self._raw = raw
        self.n_rows = n_rows
        self.row_bytes = len(raw) // n_rows

    def row(self, index: int) -> memoryview:
        """A zero-copy view of one packed row (a slice of the same buffer)."""
        start = index * self.row_bytes
        return self._raw[start : start + self.row_bytes]

    def reordered(self, order: torch.Tensor) -> memoryview:
        """A new packed buffer whose row `i` is this buffer's row `order[i]` - a copy, so it
        doesn't pin the source mmap."""
        rows = np.frombuffer(self._raw, dtype=np.uint8).reshape(self.n_rows, self.row_bytes)
        return memoryview(np.ascontiguousarray(rows[order.numpy()]))

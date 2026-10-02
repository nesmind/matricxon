import warnings

import torch
import torch.nn.functional as F
from torch import nn

from app.gguf.dequant.torch_dequant import TorchDequantizer

# Rows are dequantized in slices of at most this many elements, so a huge tensor (a 248k-row
# lm_head) never needs its whole bf16 form in VRAM at once - the packed bytes plus one slice.
_CHUNK_ELEMENTS = 64 << 20


def upload_packed(raw: memoryview, n_rows: int, device: torch.device) -> torch.Tensor:
    """`(n_rows, row_bytes)` uint8 copy of packed bytes on `device` (never aliases the mmap)."""
    with warnings.catch_warnings():  # the mmap view is read-only; we copy it right away
        warnings.simplefilter("ignore")
        flat = torch.frombuffer(raw, dtype=torch.uint8)
    return flat.reshape(n_rows, -1).to(device, copy=True)


class DevicePackedLinear(nn.Module):
    """`nn.Linear` whose weight stays quantized in device memory (`gpu_weight_mode="packed"`) and
    is dequantized with torch ops per call, in row slices. ~4x less VRAM than a bf16 weight for a
    4-bit file, at the cost of dequantizing on every forward - slower than a fused kernel, which
    this deliberately isn't. Row slices are independent (every row is whole blocks).
    """

    def __init__(
        self,
        packed: torch.Tensor,
        in_features: int,
        ggml_type: int,
        bias: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.out_features = packed.shape[0]
        self.in_features = in_features
        self._ggml_type = ggml_type
        self._dtype = dtype
        self._type_size = TorchDequantizer.geometry(ggml_type)[1]
        self.register_buffer("packed", packed, persistent=False)
        self.bias = None if bias is None else nn.Parameter(bias.to(packed.device, dtype))

    def _rows(self, packed_rows: torch.Tensor) -> torch.Tensor:
        blocks = packed_rows.reshape(-1, self._type_size)
        values = TorchDequantizer.dequantize(blocks, self._ggml_type)
        return values.reshape(packed_rows.shape[0], self.in_features).to(self._dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self._dtype)
        step = max(1, _CHUNK_ELEMENTS // self.in_features)
        if step >= self.out_features:
            return F.linear(x, self._rows(self.packed), self.bias)
        parts = [
            F.linear(x, self._rows(self.packed[start : start + step]))
            for start in range(0, self.out_features, step)
        ]
        y = torch.cat(parts, dim=-1)
        return y if self.bias is None else y + self.bias


class DevicePackedEmbedding(nn.Module):
    """Packed token-embedding table on the device: a lookup dequantizes only the rows it needs.
    `as_linear()` is the tied lm_head over the same bytes (no second copy)."""

    def __init__(
        self, packed: torch.Tensor, embedding_dim: int, ggml_type: int, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.num_embeddings = packed.shape[0]
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self._ggml_type = ggml_type
        self._type_size = TorchDequantizer.geometry(ggml_type)[1]
        self.register_buffer("packed", packed, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = self.packed[ids.reshape(-1)]
        values = TorchDequantizer.dequantize(rows.reshape(-1, self._type_size), self._ggml_type)
        return values.reshape(*ids.shape, self.embedding_dim).to(self.dtype)

    def as_linear(self) -> DevicePackedLinear:
        return DevicePackedLinear(
            self.packed, self.embedding_dim, self._ggml_type, dtype=self.dtype
        )

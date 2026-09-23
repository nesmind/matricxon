import torch
from torch import nn

from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.packed_rows import PackedRows


class QuantizedEmbedding(nn.Module):
    """`nn.Embedding` replacement that keeps `token_embd.weight` packed and dequantizes only the
    rows a forward pass actually looks up. The old path dequantized the whole table to float
    up front: for Llama-3.2-3B (128256 x 3072, Q6_K) that was ~1.5 GB of float32 and ~18 s of the
    first request's materialization, all before a single token was computed.

    A tied-embeddings model (Llama-3.2, mistral3 - no separate `output.weight`) also uses this
    same packed table as its output projection: `as_linear()` wraps the same bytes in a
    `QuantizedLinear`, so the lm_head runs on the quantized kernels instead of a float32 matmul
    against the full dequantized table (which read all 1.5 GB for every generated token).

    Like `QuantizedLinear`, `raw` must stay a live view into its source GGUF mmap until
    `release()` - see `ModelArchitecture.close`.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        ggml_type: int,
        raw: memoryview,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self._ggml_type = ggml_type
        self._raw = raw
        self._rows = PackedRows(raw, num_embeddings)
        self._strategy = QuantStrategyRegistry().get(ggml_type)

    def as_linear(self) -> QuantizedLinear:
        """The tied output projection over this same packed table (vocab x embedding_dim)."""
        return QuantizedLinear(
            self.num_embeddings, self.embedding_dim, self._ggml_type, self._raw, dtype=self.dtype
        )

    def release(self) -> None:
        """Drops the live views into the source mmap - see `QuantizedLinear.release`."""
        self._raw = None
        self._rows = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.reshape(-1).tolist()
        vectors = {
            token_id: self._strategy.dequantize(self._rows.row(token_id), self.embedding_dim)
            for token_id in set(ids)
        }
        out = torch.stack([vectors[token_id] for token_id in ids]).to(self.dtype)
        return out.reshape(*input_ids.shape, self.embedding_dim)

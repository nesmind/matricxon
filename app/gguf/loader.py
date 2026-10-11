from pathlib import Path

import torch

from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.metadata import GGUFMetadata
from app.gguf.mmap_source import GGUFMemoryMap
from app.gguf.reader import GGUFReader
from app.server.errors import UnknownModelError


class GGUFModelLoader:
    """Loads dequantized tensors from a GGUF file on demand, by name.

    Tensors are dequantized lazily, one at a time, straight into the caller's `dtype`; the raw
    memoryview into the mmap is dropped at once (a live view blocks mmap.close; GGUFMemoryMap),
    keeping peak RAM near the model's size, not 2x. A tensor an architecture never requests (e.g. a
    vision tower's weights) is never read or dequantized.
    """

    def __init__(self, path: Path, dtype: torch.dtype = torch.bfloat16) -> None:
        self._path = path
        self._dtype = dtype
        self._parsed = GGUFReader(path).read()
        self._mmap = GGUFMemoryMap(path)
        self._registry = QuantStrategyRegistry()
        self._tensor_infos_by_name = {t.name: t for t in self._parsed.tensor_infos}

    @property
    def metadata(self) -> GGUFMetadata:
        return self._parsed.metadata

    def has_tensor(self, name: str) -> bool:
        return name in self._tensor_infos_by_name

    def load_tensor(self, name: str) -> torch.Tensor:
        info = self._tensor_infos_by_name.get(name)
        if info is None:
            raise UnknownModelError(f"GGUF file has no tensor named {name!r}")

        strategy = self._registry.get(info.ggml_type)
        n_elements = info.n_elements
        byte_length = strategy.byte_length(n_elements)

        raw = self._mmap.raw_tensor_bytes(info, self._parsed.data_start_offset, byte_length)
        flat = strategy.dequantize(raw, n_elements)
        del raw

        torch_shape = tuple(reversed(info.shape))
        return flat.reshape(torch_shape).to(self._dtype)

    def raw_tensor_bytes_and_type(self, name: str) -> tuple[memoryview, int, tuple[int, ...]]:
        """The still-quantized bytes for `name` (`int(type_size)` * n_blocks, never dequantized),
        its `ggml_type` and PyTorch-order shape (`ne[]` reversed, as `load_tensor`'s reshape). For
        `QuantizedLinear`: a fused GEMV kernel needs the packed bytes, which `load_tensor` would
        defeat.

        The `memoryview` is valid only while this loader's mmap is (GGUFMemoryMap), which is why
        `ModelManager` keeps a loaded model's loader/mmap alive for its whole lifetime (the
        `_pending_loader` requirement, extended past first use).
        """
        info = self._tensor_infos_by_name.get(name)
        if info is None:
            raise UnknownModelError(f"GGUF file has no tensor named {name!r}")

        strategy = self._registry.get(info.ggml_type)
        byte_length = strategy.byte_length(info.n_elements)
        raw = self._mmap.raw_tensor_bytes(info, self._parsed.data_start_offset, byte_length)
        return raw, info.ggml_type, tuple(reversed(info.shape))

    def close(self) -> None:
        self._mmap.close()

    def __enter__(self) -> "GGUFModelLoader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

import mmap
from pathlib import Path
from types import TracebackType

from app.gguf.tensor_info import GGUFTensorInfo


class GGUFMemoryMap:
    """Zero-copy raw byte access into a GGUF file's tensor data section.

    A multi-GB file is never read into RAM as a whole - callers slice out one
    tensor's raw (still-quantized) bytes at a time via a memoryview.
    """

    def __init__(self, path: Path) -> None:
        self._file = open(path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def raw_tensor_bytes(
        self, tensor_info: GGUFTensorInfo, data_start_offset: int, byte_length: int
    ) -> memoryview:
        start = data_start_offset + tensor_info.offset
        return memoryview(self._mmap)[start : start + byte_length]

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "GGUFMemoryMap":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

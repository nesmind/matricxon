import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.gguf.constants import (
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_SUPPORTED_VERSION,
    GGUFValueType,
)
from app.gguf.metadata import GGUFMetadata
from app.gguf.tensor_info import GGUFTensorInfo
from app.server.errors import InvalidGGUFError

_SCALAR_STRUCTS: dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
}


@dataclass(frozen=True)
class ParsedGGUF:
    version: int
    metadata: GGUFMetadata
    tensor_infos: list[GGUFTensorInfo]
    data_start_offset: int


class GGUFReader:
    """Parses a GGUF file's header, metadata, and tensor-info table.

    Only reads these (KB-scale) eagerly; tensor data bytes are never touched
    here - see GGUFMemoryMap for zero-copy access to those.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def read(self) -> ParsedGGUF:
        with open(self._path, "rb") as f:
            self._check_magic(f)
            version = self._read_u32(f)
            if version != GGUF_SUPPORTED_VERSION:
                raise InvalidGGUFError(f"Unsupported GGUF version: {version}")
            tensor_count = self._read_u64(f)
            kv_count = self._read_u64(f)

            metadata = GGUFMetadata(self._read_metadata_kvs(f, kv_count))
            tensor_infos = self._read_tensor_infos(f, tensor_count)

            alignment = metadata.get_u32("general.alignment", GGUF_DEFAULT_ALIGNMENT)
            data_start_offset = self._align(f.tell(), alignment)

        return ParsedGGUF(
            version=version,
            metadata=metadata,
            tensor_infos=tensor_infos,
            data_start_offset=data_start_offset,
        )

    def _check_magic(self, f: BinaryIO) -> None:
        magic = f.read(len(GGUF_MAGIC))
        if magic != GGUF_MAGIC:
            raise InvalidGGUFError(f"Not a GGUF file (bad magic: {magic!r})")

    def _read_metadata_kvs(self, f: BinaryIO, count: int) -> dict[str, object]:
        values: dict[str, object] = {}
        for _ in range(count):
            key = self._read_str(f)
            value_type = GGUFValueType(self._read_u32(f))
            values[key] = self._read_value(f, value_type)
        return values

    def _read_tensor_infos(self, f: BinaryIO, count: int) -> list[GGUFTensorInfo]:
        infos = []
        for _ in range(count):
            name = self._read_str(f)
            n_dims = self._read_u32(f)
            shape = tuple(self._read_u64(f) for _ in range(n_dims))
            ggml_type = self._read_u32(f)
            offset = self._read_u64(f)
            infos.append(GGUFTensorInfo(name=name, shape=shape, ggml_type=ggml_type, offset=offset))
        return infos

    def _read_value(self, f: BinaryIO, value_type: GGUFValueType) -> object:
        if value_type == GGUFValueType.STRING:
            return self._read_str(f)
        if value_type == GGUFValueType.BOOL:
            return self._read_u8(f) != 0
        if value_type == GGUFValueType.ARRAY:
            elem_type = GGUFValueType(self._read_u32(f))
            count = self._read_u64(f)
            return [self._read_value(f, elem_type) for _ in range(count)]
        return self._read_scalar(f, value_type)

    def _read_scalar(self, f: BinaryIO, value_type: GGUFValueType) -> int | float:
        fmt = _SCALAR_STRUCTS[value_type]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def _read_str(self, f: BinaryIO) -> str:
        length = self._read_u64(f)
        return f.read(length).decode("utf-8")

    def _read_u8(self, f: BinaryIO) -> int:
        return struct.unpack("<B", f.read(1))[0]

    def _read_u32(self, f: BinaryIO) -> int:
        return struct.unpack("<I", f.read(4))[0]

    def _read_u64(self, f: BinaryIO) -> int:
        return struct.unpack("<Q", f.read(8))[0]

    @staticmethod
    def _align(offset: int, alignment: int) -> int:
        return ((offset + alignment - 1) // alignment) * alignment

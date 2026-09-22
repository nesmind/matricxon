"""Dev/test tool: hand-builds tiny synthetic GGUF v3 byte streams.

Used by tests/unit/test_gguf_reader.py and test_dequant_kernels.py to exercise
GGUFReader/GGUFMemoryMap against known-good bytes without needing a real model
file. Not used anywhere in the app itself.
"""

import struct
from pathlib import Path

from app.gguf.constants import GGUFValueType

_SCALAR_PACK_FORMATS: dict[GGUFValueType, str] = {
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


class GGUFBuilder:
    """Fluent builder for a minimal valid GGUF v3 byte stream."""

    def __init__(self, alignment: int = 32) -> None:
        self._alignment = alignment
        self._metadata_entries: list[bytes] = []
        self._tensors: list[tuple[str, list[int], int, bytes]] = []

    def set_str(self, key: str, value: str) -> "GGUFBuilder":
        return self._set(key, GGUFValueType.STRING, value)

    def set_u32(self, key: str, value: int) -> "GGUFBuilder":
        return self._set(key, GGUFValueType.UINT32, value)

    def set_f32(self, key: str, value: float) -> "GGUFBuilder":
        return self._set(key, GGUFValueType.FLOAT32, value)

    def set_bool(self, key: str, value: bool) -> "GGUFBuilder":
        return self._set(key, GGUFValueType.BOOL, value)

    def set_array(self, key: str, elem_type: GGUFValueType, values: list) -> "GGUFBuilder":
        body = struct.pack("<I", int(elem_type)) + struct.pack("<Q", len(values))
        for value in values:
            body += self._encode_value(elem_type, value, tagged=False)
        self._metadata_entries.append(
            self._encode_str(key) + struct.pack("<I", int(GGUFValueType.ARRAY)) + body
        )
        return self

    def add_tensor(
        self, name: str, shape: list[int], ggml_type: int, raw_bytes: bytes
    ) -> "GGUFBuilder":
        self._tensors.append((name, shape, ggml_type, raw_bytes))
        return self

    def build(self) -> bytes:
        header = (
            b"GGUF"
            + struct.pack("<I", 3)
            + struct.pack("<Q", len(self._tensors))
            + struct.pack("<Q", len(self._metadata_entries))
        )
        metadata_bytes = b"".join(self._metadata_entries)
        tensor_info_bytes = self._build_tensor_info_table()

        pre_data = header + metadata_bytes + tensor_info_bytes
        pre_data += b"\x00" * (self._align(len(pre_data)) - len(pre_data))

        data = b"".join(
            raw + b"\x00" * (self._align(len(raw)) - len(raw)) for _, _, _, raw in self._tensors
        )
        return pre_data + data

    def write(self, path: Path) -> Path:
        path.write_bytes(self.build())
        return path

    def _build_tensor_info_table(self) -> bytes:
        offsets = []
        offset = 0
        for _, _, _, raw in self._tensors:
            offsets.append(offset)
            offset += self._align(len(raw))

        table = b""
        for (name, shape, ggml_type, _), tensor_offset in zip(self._tensors, offsets, strict=True):
            table += self._encode_str(name)
            table += struct.pack("<I", len(shape))
            for dim in shape:
                table += struct.pack("<Q", dim)
            table += struct.pack("<I", ggml_type)
            table += struct.pack("<Q", tensor_offset)
        return table

    def _set(self, key: str, value_type: GGUFValueType, value: object) -> "GGUFBuilder":
        self._metadata_entries.append(self._encode_str(key) + self._encode_value(value_type, value))
        return self

    def _align(self, length: int) -> int:
        return ((length + self._alignment - 1) // self._alignment) * self._alignment

    @staticmethod
    def _encode_str(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def _encode_value(self, value_type: GGUFValueType, value: object, tagged: bool = True) -> bytes:
        if value_type == GGUFValueType.STRING:
            body = self._encode_str(value)
        elif value_type == GGUFValueType.BOOL:
            body = struct.pack("<B", 1 if value else 0)
        else:
            body = struct.pack(_SCALAR_PACK_FORMATS[value_type], value)
        return (struct.pack("<I", int(value_type)) + body) if tagged else body

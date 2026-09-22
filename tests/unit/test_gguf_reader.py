import struct
from pathlib import Path

import pytest

from app.gguf.constants import GGUFValueType
from app.gguf.mmap_source import GGUFMemoryMap
from app.gguf.reader import GGUFReader
from app.server.errors import InvalidGGUFError
from scripts.make_tiny_gguf import GGUFBuilder


def _write_minimal(path: Path) -> None:
    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    (
        GGUFBuilder()
        .set_str("general.architecture", "test")
        .set_u32("test.block_count", 2)
        .set_f32("test.rope.freq_base", 10000.5)
        .set_bool("test.attention.causal", True)
        .set_array("test.some_ints", GGUFValueType.UINT32, [1, 2, 3])
        .add_tensor("weight", [4], ggml_type=0, raw_bytes=raw)
        .write(path)
    )


def test_reads_header_and_scalar_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_minimal(path)

    parsed = GGUFReader(path).read()

    assert parsed.version == 3
    assert parsed.metadata.architecture == "test"
    assert parsed.metadata.get_u32("test.block_count") == 2
    assert parsed.metadata.get_f32("test.rope.freq_base") == pytest.approx(10000.5, rel=1e-5)
    assert parsed.metadata.get_bool("test.attention.causal") is True
    assert parsed.metadata.get_array("test.some_ints") == [1, 2, 3]


def test_reads_tensor_info(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_minimal(path)

    parsed = GGUFReader(path).read()

    [tensor] = parsed.tensor_infos
    assert tensor.name == "weight"
    assert tensor.shape == (4,)
    assert tensor.ggml_type == 0
    assert tensor.n_elements == 4


def test_data_start_offset_is_aligned(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_minimal(path)

    parsed = GGUFReader(path).read()

    assert parsed.data_start_offset % 32 == 0


def test_mmap_reads_back_correct_tensor_bytes(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_minimal(path)

    parsed = GGUFReader(path).read()
    [tensor] = parsed.tensor_infos

    with GGUFMemoryMap(path) as mmap_source:
        raw = mmap_source.raw_tensor_bytes(tensor, parsed.data_start_offset, byte_length=16)
        values = struct.unpack("<4f", bytes(raw))
        del raw  # a live memoryview slice blocks mmap.close() - see GGUFMemoryMap.close()

    assert values == (1.0, 2.0, 3.0, 4.0)


def test_bad_magic_raises_invalid_gguf_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 16)

    with pytest.raises(InvalidGGUFError):
        GGUFReader(path).read()

"""Loads the grid/sign lookup tables for the grid-based I-quants (iq2_family.py, iq3_family.py,
iq1_family.py). The data is extracted verbatim from ggml's source (ggml/src/ggml-common.h) by
tools/extract_iq_grids.py (not hand-transcribed: an error in a multi-thousand-entry table would go
unnoticed) and checked in as .npy files under app/gguf/dequant/data/.

Each grid entry (packed uint64 or uint32) is reinterpreted as N signed int8 via `.view(np.int8)`,
NumPy's equivalent of ggml's `(const int8_t *)(grid + idx)` cast (each dequantize_row_iq*'s C source
is quoted in the strategies).
"""

from functools import cache
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).resolve().parent / "data"


@cache
def _load(name: str) -> np.ndarray:
    return np.load(_DATA_DIR / f"{name}.npy")


def kmask_iq2xs() -> np.ndarray:
    """8 bytes: 1, 2, 4, 8, 16, 32, 64, 128 - per-lane sign bit masks."""
    return _load("kmask_iq2xs")


def ksigns_iq2xs() -> np.ndarray:
    """128 bytes - precomputed sign-parity bytes, indexed by a 7-bit sign selector."""
    return _load("ksigns_iq2xs")


def iq2xxs_grid_i8() -> np.ndarray:
    """(256, 8) int8 - each of the 256 uint64 grid entries reinterpreted as 8 signed bytes."""
    return _load("iq2xxs_grid").view(np.int8).reshape(-1, 8)


def iq2xs_grid_i8() -> np.ndarray:
    """(512, 8) int8."""
    return _load("iq2xs_grid").view(np.int8).reshape(-1, 8)


def iq2s_grid_i8() -> np.ndarray:
    """(1024, 8) int8."""
    return _load("iq2s_grid").view(np.int8).reshape(-1, 8)


def iq3xxs_grid_i8() -> np.ndarray:
    """(256, 4) int8 - iq3xxs_grid is uint32-packed, so 4 signed bytes per entry, not 8."""
    return _load("iq3xxs_grid").view(np.int8).reshape(-1, 4)


def iq3s_grid_i8() -> np.ndarray:
    """(512, 4) int8."""
    return _load("iq3s_grid").view(np.int8).reshape(-1, 4)


def iq1s_grid_i8() -> np.ndarray:
    """(2048, 8) int8 - shared by both IQ1_S and IQ1_M."""
    return _load("iq1s_grid").view(np.int8).reshape(-1, 8)

#!/usr/bin/env python3
"""One-off extraction of the I-quant lookup tables (`iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`,
`iq3xxs_grid`, `iq3s_grid`, `iq1s_grid`) and sign tables (`kmask_iq2xs`, `ksigns_iq2xs`) for
iq2_family.py, iq3_family.py and iq1_family.py, taken from ggml's source (not hand-transcribed).

Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-common.h (pulled).

Re-run on a fresh ggml-common.h to re-sync or re-verify the.npy files in app/gguf/dequant/data/.

Usage: python3 tools/extract_iq_grids.py <path-to-ggml-common.h>
"""

import re
import sys
from pathlib import Path

import numpy as np

_DTYPES = {"uint8_t": "<u1", "uint32_t": "<u4", "uint64_t": "<u8"}

# {C table name: declared entry count}, to assert the parsed count matches the source.
_EXPECTED = {
    "kmask_iq2xs": 8,
    "ksigns_iq2xs": 128,
    "iq2xxs_grid": 256,
    "iq2xs_grid": 512,
    "iq2s_grid": 1024,
    "iq3xxs_grid": 256,
    "iq3s_grid": 512,
    "iq1s_grid": 2048,
}


def _extract_table(source: str, c_type: str, name: str) -> np.ndarray:
    pattern = re.compile(
        rf"GGML_TABLE_BEGIN\({re.escape(c_type)},\s*{re.escape(name)},\s*[\w]+\)(.*?)GGML_TABLE_END\(\)",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        raise ValueError(f"Could not find table {name!r} of type {c_type!r} in source")
    literals = [tok.strip() for tok in match.group(1).split(",")]
    literals = [tok for tok in literals if tok]
    values = [int(tok, 0) for tok in literals]
    expected = _EXPECTED[name]
    if len(values) != expected:
        raise ValueError(f"{name}: parsed {len(values)} entries, expected {expected}")
    return np.array(values, dtype=_DTYPES[c_type])


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    source = Path(sys.argv[1]).read_text()
    out_dir = Path(__file__).resolve().parent.parent / "app" / "gguf" / "dequant" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = [
        ("uint8_t", "kmask_iq2xs"),
        ("uint8_t", "ksigns_iq2xs"),
        ("uint64_t", "iq2xxs_grid"),
        ("uint64_t", "iq2xs_grid"),
        ("uint64_t", "iq2s_grid"),
        ("uint32_t", "iq3xxs_grid"),
        ("uint32_t", "iq3s_grid"),
        ("uint64_t", "iq1s_grid"),
    ]
    for c_type, name in tables:
        arr = _extract_table(source, c_type, name)
        out_path = out_dir / f"{name}.npy"
        np.save(out_path, arr)
        print(f"{name}: {len(arr)} entries ({arr.dtype}) -> {out_path}")


if __name__ == "__main__":
    main()

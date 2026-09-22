"""Shared plumbing for the M3 HF-oracle cross-check (see scripts/README_oracle.md).

Both dump scripts run as separate, memory-capped OS processes (never loading
matricxon's model and the HF oracle model at once) and only ever materialize
the first `N_LAYERS` decoder layers plus the embedding/final-norm - never the
full 26-layer, vision-carrying checkpoint - which is what keeps this
tractable on a machine with ~15GB RAM and no GPU.
"""

import json
import struct
from pathlib import Path

import httpx
import torch

HF_REPO = "mistralai/Ministral-3-3B-Instruct-2512"
# Updated 2026-09-19 - pAIring re-pulled ministral-3:3b under a new content
# hash (see tests/integration/conftest.py's REAL_MINISTRAL_BLOB for details).
DEFAULT_GGUF_PATH = Path(
    "/home/home/Code/Py/AI/pAIring/models/blobs/"
    "sha256-9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8"
)
DEFAULT_N_LAYERS = 4
PROMPT = "The capital of France is"
DUMP_DIR = Path(__file__).resolve().parents[2] / "data" / "oracle"


class RemoteSafetensorsReader:
    """Fetches individual tensors from a Hub safetensors file by HTTP Range request.

    The checkpoint is 4.7GB and only ~1% of it (a handful of early-layer
    tensors) is ever needed here, so this never downloads the file - it reads
    the safetensors header (a small JSON blob) once, then issues one ranged
    GET per requested tensor.
    """

    def __init__(self, repo: str, filename: str = "model.safetensors") -> None:
        self._url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        self._client = httpx.Client(follow_redirects=True, timeout=60.0)
        header_len = struct.unpack("<Q", self._ranged_get(0, 7))[0]
        self._header: dict = json.loads(self._ranged_get(8, 8 + header_len - 1))
        self._data_start = 8 + header_len

    def _ranged_get(self, start: int, end_inclusive: int) -> bytes:
        resp = self._client.get(self._url, headers={"Range": f"bytes={start}-{end_inclusive}"})
        resp.raise_for_status()
        return resp.content

    def close(self) -> None:
        self._client.close()

    def get_dequantized(self, name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Fetches `name`, dequantizing fp8 weights via their sibling `*_scale_inv`."""
        entry = self._header[name]
        raw = self._fetch_raw(entry)

        if entry["dtype"] == "F8_E4M3":
            scale = self._fetch_raw(self._header[name + "_scale_inv"]).to(torch.float32)
            weight_t = raw.to(torch.float32) * scale
            return weight_t.to(dtype)

        return raw.to(dtype)

    def _fetch_raw(self, entry: dict) -> torch.Tensor:
        start, end = entry["data_offsets"]
        torch_dtype = _SAFETENSORS_DTYPES[entry["dtype"]]
        if start == end:
            return torch.zeros((), dtype=torch_dtype)
        body = self._ranged_get(self._data_start + start, self._data_start + end - 1)
        flat = torch.frombuffer(bytearray(body), dtype=torch_dtype)
        return flat.reshape(entry["shape"])


_SAFETENSORS_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
}


def build_input_ids(repo: str = HF_REPO) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo)
    return tokenizer(PROMPT, return_tensors="pt")["input_ids"]

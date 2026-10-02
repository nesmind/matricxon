"""A whole decode (prefill, then token-by-token through the KV cache) with the fused ops switched
on must give the same logits as without them - for every architecture's attention/norm/rope wiring,
on both backends. Tiny fixtures (tests/unit/test_batched_decode.py's cases)."""

from pathlib import Path

import pytest
import torch

from app.gguf.loader import GGUFModelLoader
from app.native.fused_ops import FusedOps, _NativeBackend, _NumbaBackend
from app.native.library import NativeKernelLibrary
from tests.unit.test_batched_decode import CASES

PROMPT = [3, 40, 77, 120, 9, 200]
NEXT_TOKENS = [11, 12, 13]


def _decode_logits(model, steps: list[list[int]]) -> list[torch.Tensor]:
    cache = model.build_cache(max_seq_len=64, dtype=torch.float32)
    out = []
    position = 0
    with torch.no_grad():
        for ids in steps:
            positions = torch.arange(position, position + len(ids))
            out.append(model(torch.tensor([ids]), cache, positions, last_logits_only=True))
            cache.advance(len(ids))
            position += len(ids)
    return out


@pytest.mark.parametrize("backend", ["native", "numba"])
@pytest.mark.parametrize(("name", "builder", "cls"), CASES, ids=[c[0] for c in CASES])
def test_fused_decode_matches_plain_torch(
    name: str, builder, cls, backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = cls.from_gguf(GGUFModelLoader(builder(tmp_path / "m.gguf"), dtype=torch.float32))
    model.eval()
    steps = [PROMPT, *([t] for t in NEXT_TOKENS)]
    monkeypatch.setattr(FusedOps, "_active", None)
    expected = _decode_logits(model, steps)

    chosen = (
        _NativeBackend(NativeKernelLibrary().load(), 2)
        if backend == "native"
        else _NumbaBackend(attention=True)
    )
    monkeypatch.setattr(FusedOps, "_active", FusedOps(chosen))
    actual = _decode_logits(model, steps)
    for want, got in zip(expected, actual, strict=True):
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), f"{name}/{backend}"

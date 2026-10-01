"""A decode step run as part of a batch must give the same logits as run alone - checked per
architecture before it may set `SUPPORTS_BATCHED_DECODE`. Three sequences of different lengths
(so padding/masks matter), a few steps each, tiny fixtures."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from app.architectures.base import ModelArchitecture
from app.architectures.command_r import CommandRArchitecture
from app.architectures.falcon import FalconArchitecture
from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.granite import GraniteArchitecture
from app.architectures.granitemoe import GraniteMoeArchitecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.mistral3 import Mistral3TextArchitecture
from app.architectures.nemotron_h import NemotronHArchitecture
from app.architectures.phi2 import Phi2Architecture
from app.architectures.qwen2 import Qwen2Architecture
from app.architectures.qwen3 import Qwen3Architecture
from app.architectures.qwen35 import Qwen35Architecture
from app.architectures.starcoder2 import Starcoder2Architecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.batch_decode import BatchDecoder, DecodeStep
from tests.tiny_gguf import build_tiny_mistral3_gguf
from tests.tiny_gguf_command_r import build_tiny_command_r_gguf
from tests.tiny_gguf_falcon import build_tiny_falcon_gguf
from tests.tiny_gguf_gemma4 import (
    build_tiny_gemma4_gguf,
    build_tiny_gemma4_moe_gguf,
    build_tiny_gemma4_ple_kv_shared_gguf,
)
from tests.tiny_gguf_granite import build_tiny_granite_gguf
from tests.tiny_gguf_granitemoe import build_tiny_granitemoe_gguf
from tests.tiny_gguf_llama import build_tiny_llama_gguf
from tests.tiny_gguf_llama_moe import build_tiny_llama_moe_gguf
from tests.tiny_gguf_nemotron_h import build_tiny_nemotron_h_gguf
from tests.tiny_gguf_phi2 import build_tiny_phi2_gguf
from tests.tiny_gguf_qwen2 import build_tiny_qwen2_gguf
from tests.tiny_gguf_qwen3 import build_tiny_qwen3_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf
from tests.tiny_gguf_starcoder2 import build_tiny_starcoder2_gguf

CASES: list[tuple[str, Callable[[Path], Path], type[ModelArchitecture]]] = [
    ("mistral3", build_tiny_mistral3_gguf, Mistral3TextArchitecture),
    ("llama", build_tiny_llama_gguf, LlamaArchitecture),
    ("llama_moe", build_tiny_llama_moe_gguf, LlamaArchitecture),
    ("qwen2", build_tiny_qwen2_gguf, Qwen2Architecture),
    ("qwen3", build_tiny_qwen3_gguf, Qwen3Architecture),
    ("qwen35", build_tiny_qwen35_gguf, Qwen35Architecture),
    ("granite", build_tiny_granite_gguf, GraniteArchitecture),
    ("granitemoe", build_tiny_granitemoe_gguf, GraniteMoeArchitecture),
    ("command_r", build_tiny_command_r_gguf, CommandRArchitecture),
    ("phi2", build_tiny_phi2_gguf, Phi2Architecture),
    ("starcoder2", build_tiny_starcoder2_gguf, Starcoder2Architecture),
    ("falcon", build_tiny_falcon_gguf, FalconArchitecture),
    ("gemma4", build_tiny_gemma4_gguf, Gemma4Architecture),
    ("gemma4_ple_kv_shared", build_tiny_gemma4_ple_kv_shared_gguf, Gemma4Architecture),
    ("gemma4_moe", build_tiny_gemma4_moe_gguf, Gemma4Architecture),
    ("nemotron_h", build_tiny_nemotron_h_gguf, NemotronHArchitecture),
]
PROMPTS = [[3, 40, 77, 120, 9, 200], [5, 6, 7, 8, 9, 10, 11, 12, 13], [60, 61, 62, 63]]
STEPS = 3


def _prefilled(model: ModelArchitecture) -> list[object]:
    caches = []
    for ids in PROMPTS:
        cache = model.build_cache(max_seq_len=64, dtype=torch.float32)
        model(torch.tensor([ids]), cache, torch.arange(len(ids)), last_logits_only=True)
        cache.advance(len(ids))
        caches.append(cache)
    return caches


def _decode_logits(model: ModelArchitecture, batched: bool) -> list[list[torch.Tensor]]:
    caches = _prefilled(model)
    decoder = BatchDecoder(model)
    decoder._can_batch = batched  # discovery: force the path under test
    out = []
    for step in range(STEPS):
        steps = [
            DecodeStep(3 + (7 * step + 11 * b) % 200, cache, cache.length)
            for b, cache in enumerate(caches)
        ]
        results = decoder.decode(steps) if batched else [decoder.decode_alone(s) for s in steps]
        for cache in caches:
            cache.advance(1)
        out.append(results)
    return out


def batched_matches(model: ModelArchitecture) -> bool:
    alone, together = _decode_logits(model, False), _decode_logits(model, True)
    return all(
        isinstance(a, torch.Tensor)
        and isinstance(t, torch.Tensor)
        and torch.allclose(a, t, atol=1e-4)
        for row_a, row_t in zip(alone, together, strict=True)
        for a, t in zip(row_a, row_t, strict=True)
    )


def _load(builder: Callable[[Path], Path], cls: type[ModelArchitecture], tmp_path: Path):
    path = builder(tmp_path / "m.gguf")
    return cls.from_gguf(GGUFModelLoader(path, dtype=torch.float32)).eval()


@pytest.mark.parametrize(("name", "builder", "cls"), CASES, ids=[c[0] for c in CASES])
def test_a_batched_decode_step_matches_decoding_alone(
    name: str, builder: Callable[[Path], Path], cls: type[ModelArchitecture], tmp_path: Path
) -> None:
    model = _load(builder, cls, tmp_path)
    matches = batched_matches(model)
    if model.SUPPORTS_BATCHED_DECODE:  # per model: a MoE checkpoint of a flagged class turns it off
        assert matches, f"{cls.__name__} ({name}) claims batched decode but its logits differ"
    print(f"{name}: batched matches = {matches}, flagged = {model.SUPPORTS_BATCHED_DECODE}")
    model.close()

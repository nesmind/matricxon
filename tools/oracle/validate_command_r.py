"""Real-weight validation of `CommandRArchitecture` against `CohereLabs/aya-23-8B` (same
`CohereForCausalLM`/`command-r` GGUF architecture as Command-R; no small real Command-R exists).
Downloads the community GGUF once (~5GB).

Unlike `validate_qwen2.py`/`validate_qwen3.py`, no HF layer comparison: the safetensors repo is
gated (401 without `HF_TOKEN`) and its ~16GB bf16 weights are too tight for 15GB RAM.
Instead it runs an end-to-end generation coherence check, the signal that caught the
`unpermute_rope_rows` bug for `qwen2`/`qwen3`. The "no permutation" and "bias-free LayerNorm"
formulas are also backed by source evidence (see `CommandRArchitecture`).
"""

from pathlib import Path

import httpx
import torch

from app.architectures.registry import ArchitectureRegistry
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.tokenizer import GGUFTokenizer

GGUF_REPO = "bartowski/aya-23-8B-GGUF"
GGUF_FILENAME = "aya-23-8B-Q4_K_M.gguf"
GGUF_PATH = Path(__file__).resolve().parents[2] / "data" / "oracle" / GGUF_FILENAME
PROMPTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "2 + 2 =",
]


def _ensure_gguf_downloaded() -> None:
    if GGUF_PATH.exists():
        return
    GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{GGUF_REPO}/resolve/main/{GGUF_FILENAME}"
    print(f"Downloading {url} -> {GGUF_PATH} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
        response.raise_for_status()
        with GGUF_PATH.open("wb") as f:
            for chunk in response.iter_bytes(4 * 1024 * 1024):
                f.write(chunk)
    print(f"Downloaded {GGUF_PATH.stat().st_size} bytes.")


def main() -> None:
    _ensure_gguf_downloaded()

    loader = GGUFModelLoader(str(GGUF_PATH), dtype=torch.bfloat16)
    model = ArchitectureRegistry().resolve(loader.metadata).from_gguf(loader)
    tokenizer = GGUFTokenizer(loader.metadata)
    model.eval()
    print(f"logit_scale={model.logit_scale}, tied_embeddings={model._tied_embeddings}")

    engine = ChatEngine(model, eos_token_ids={tokenizer.eos_token_id})
    for prompt in PROMPTS:
        input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
        sampling = SamplingConfig(temperature=0.0, num_ctx=64, num_predict=20)
        req = GenerationRequest(input_ids=input_ids, sampling=sampling)
        tokens = list(engine.stream(req))
        text = "".join(tokenizer.decode([t]) for t in tokens)
        print(f"{prompt!r} -> {text!r}")


if __name__ == "__main__":
    main()

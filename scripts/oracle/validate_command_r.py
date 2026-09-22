"""Real-weight validation for `CommandRArchitecture` against `CohereLabs/aya-23-8B` - a real
Cohere-released 8B model on the exact same `CohereForCausalLM`/`command-r` GGUF architecture as
classic Command-R (no small enough real Command-R checkpoint exists at all - see
`CommandRArchitecture`'s own docstring). Downloads the real community GGUF once (~5GB, cached
after) if not already present locally.

Unlike `validate_qwen2.py`/`validate_qwen3.py`, this does **not** do a full layer-by-layer HF
`transformers` comparison: `CohereLabs/aya-23-8B`'s real safetensors repo is gated (confirmed:
401 without an `HF_TOKEN`, 2026-09-22) and its bf16 weights are ~16GB on their own - too tight
for a 15GB-RAM single-process comparison even if access were available. Instead this runs a real
end-to-end generation coherence check - the same signal that caught the real `unpermute_rope_rows`
bug for `qwen2`/`qwen3` well before any HF-side layer comparison was needed. The "no permutation
needed" and "bias-free LayerNorm" formulas here are additionally backed by direct source evidence
(see `CommandRArchitecture`'s own docstring) rather than resting on this check alone.
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

"""Cross-checks `Qwen3Architecture`'s forward pass against the real HF `Qwen3-0.6B` model -
`validate_bert.py`-style single-process comparison. Downloads the real small GGUF once
(~397MB, cached after) if not already present locally; the HF reference downloads/caches itself
via `transformers`.

This is the real target of this session's real-weight oracle validation: whether QK-norm is
wired correctly (placement, epsilon, and whether its weight needs `unpermute_rope_rows` - see
`Qwen3Architecture`'s own docstring for why the answer turned out to be "no," the same as its
`qwen2` sibling, contrary to the original reasoning).
"""

from pathlib import Path

import httpx
import torch
from transformers import AutoModelForCausalLM

from app.architectures.registry import ArchitectureRegistry
from app.gguf.loader import GGUFModelLoader
from app.runtime.tokenizer import GGUFTokenizer

GGUF_REPO = "unsloth/Qwen3-0.6B-GGUF"
GGUF_FILENAME = "Qwen3-0.6B-Q4_K_M.gguf"
HF_REPO = "Qwen/Qwen3-0.6B"
GGUF_PATH = Path(__file__).resolve().parents[2] / "data" / "oracle" / GGUF_FILENAME
PROMPT = "The capital of France is"
COSINE_THRESHOLD = 0.98  # Q4_K_M vs bf16 - quantization noise, not machine precision


def _ensure_gguf_downloaded() -> None:
    if GGUF_PATH.exists():
        return
    GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{GGUF_REPO}/resolve/main/{GGUF_FILENAME}"
    print(f"Downloading {url} -> {GGUF_PATH} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with GGUF_PATH.open("wb") as f:
            for chunk in response.iter_bytes(1024 * 1024):
                f.write(chunk)
    print(f"Downloaded {GGUF_PATH.stat().st_size} bytes.")


def main() -> None:
    _ensure_gguf_downloaded()

    loader = GGUFModelLoader(str(GGUF_PATH), dtype=torch.float32)
    model = ArchitectureRegistry().resolve(loader.metadata).from_gguf(loader)
    tokenizer = GGUFTokenizer(loader.metadata)
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(PROMPT, add_bos=False)], dtype=torch.long)
    print(f"Prompt tokenized to {input_ids.shape[1]} tokens (matricxon's own tokenizer).")

    with torch.no_grad():
        model.forward(input_ids)  # trigger real materialization before manual replay below

    hf_model = AutoModelForCausalLM.from_pretrained(HF_REPO, dtype=torch.float32).eval()
    hf_hidden_states: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(hf_model.model.layers):
        layer.register_forward_hook(
            lambda _m, _i, out, i=i: hf_hidden_states.__setitem__(
                f"layer_{i}", (out[0] if isinstance(out, tuple) else out)
            )
        )

    with torch.no_grad():
        ours_logits = model.forward(input_ids)
        hf_logits = hf_model(input_ids, use_cache=False).logits

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    all_ok = True
    with torch.no_grad():
        cos, sin = model.rope(torch.arange(input_ids.shape[1], dtype=torch.long))
        x = model.token_embd(input_ids)
        for i, layer in enumerate(model.layers):
            x = layer(x, cos, sin)
            c = cosine(x, hf_hidden_states[f"layer_{i}"])
            ok = c >= COSINE_THRESHOLD
            all_ok &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] layer_{i}: cosine={c:.6f}")

    final_cosine = cosine(ours_logits, hf_logits)
    ok = final_cosine >= COSINE_THRESHOLD
    all_ok &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] final logits: cosine={final_cosine:.6f}")

    print("\nAll matched." if all_ok else "\nMismatches found - see FAIL lines above.")


if __name__ == "__main__":
    main()

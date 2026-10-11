"""Cross-checks NomicBertArchitecture's forward pass against HF `nomic-ai/nomic-embed-text-v1.5`
(custom `trust_remote_code=True` code; needs dev-only `einops`). No memory cap: ~137M params, F16
GGUF."""

from pathlib import Path

import torch
from transformers import AutoModel

from app.architectures.registry import ArchitectureRegistry
from app.gguf.loader import GGUFModelLoader
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer

GGUF_PATH = Path(
    "/home/home/Code/Py/AI/pAIring/models/blobs/"
    "sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6"
)
HF_REPO = "nomic-ai/nomic-embed-text-v1.5"
PROMPT = "search_query: what is the capital of France?"
COSINE_THRESHOLD = 0.999


def main() -> None:
    # No `with` block: weights are read at the first forward pass, which closes the loader
    # (ModelArchitecture._ensure_materialized).
    loader = GGUFModelLoader(str(GGUF_PATH), dtype=torch.float32)
    model = ArchitectureRegistry().resolve(loader.metadata).from_gguf(loader)
    tokenizer = WordPieceTokenizer(loader.metadata)
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)

    hf_model = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True).eval()
    # The remote-code class calls `get_extended_attention_mask`, removed in this transformers
    # version (a reference-side incompatibility). Patch in the old implementation; with no padding
    # it is a no-op (all-zeros mask).
    hf_model.get_extended_attention_mask = lambda mask, _shape: (
        (1.0 - mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
    )
    hf_hidden_states = {}
    for i, layer in enumerate(hf_model.encoder.layers):
        layer.register_forward_hook(
            lambda _m, _i, out, i=i: hf_hidden_states.__setitem__(
                f"layer_{i}", out[0] if isinstance(out, tuple) else out
            )
        )

    with torch.no_grad():
        ours_final = model.forward(input_ids)
        attention_mask = torch.ones_like(input_ids)
        hf_final = hf_model(input_ids, attention_mask=attention_mask).last_hidden_state

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    final_cosine = cosine(ours_final, hf_final)
    all_ok = final_cosine >= COSINE_THRESHOLD
    print(f"[{'PASS' if all_ok else 'FAIL'}] final hidden state: cosine={final_cosine:.6f}")

    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1])
        token_type_ids = torch.zeros_like(input_ids)
        cos, sin = model.rope(position_ids)
        x = model.token_embd(input_ids) + model.token_types(token_type_ids)
        x = model.token_embd_norm(x)
        for i, layer in enumerate(model.layers):
            x = layer(x, cos, sin)
            c = cosine(x, hf_hidden_states[f"layer_{i}"])
            ok = c >= COSINE_THRESHOLD
            all_ok &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] layer_{i}: cosine={c:.6f}")

    print("\nAll matched." if all_ok else "\nMismatches found - see FAIL lines above.")


if __name__ == "__main__":
    main()

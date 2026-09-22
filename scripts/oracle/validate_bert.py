"""Cross-checks BertArchitecture's forward pass against the real HF
`sentence-transformers/all-MiniLM-L6-v2` model - the encoder equivalent of
run_m3_oracle_check.sh. No memory cap needed: this whole model is ~22M
params, F16 in the GGUF - loads trivially in float32 either side.
"""

from pathlib import Path

import torch
from transformers import AutoModel

from app.architectures.registry import ArchitectureRegistry
from app.gguf.loader import GGUFModelLoader
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer

GGUF_PATH = Path(
    "/home/home/Code/Py/AI/pAIring/models/blobs/"
    "sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662"
)
HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT = "The quick brown fox jumps over the lazy dog."
COSINE_THRESHOLD = 0.999  # F16-quantized both sides - should be near machine precision


def main() -> None:
    # No `with` block (M10's on-the-fly dequant): real weight bytes aren't
    # read until this model's first forward pass, closed automatically by
    # that call instead - see ModelArchitecture._ensure_materialized.
    loader = GGUFModelLoader(str(GGUF_PATH), dtype=torch.float32)
    model = ArchitectureRegistry().resolve(loader.metadata).from_gguf(loader)
    tokenizer = WordPieceTokenizer(loader.metadata)
    model.eval()

    input_ids = torch.tensor([tokenizer.encode(PROMPT)], dtype=torch.long)

    hf_model = AutoModel.from_pretrained(HF_REPO).eval()
    hf_hidden_states = {}
    for i, layer in enumerate(hf_model.encoder.layer):
        layer.register_forward_hook(
            lambda _m, _i, out, i=i: hf_hidden_states.__setitem__(f"layer_{i}", out[0])
        )

    with torch.no_grad():
        ours_final = model.forward(input_ids)
        hf_final = hf_model(input_ids).last_hidden_state

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

    final_cosine = cosine(ours_final, hf_final)
    all_ok = final_cosine >= COSINE_THRESHOLD
    print(f"[{'PASS' if all_ok else 'FAIL'}] final hidden state: cosine={final_cosine:.6f}")

    # Per-layer spot check: replay our own embedding+layer stack manually so
    # each layer's output can be compared against HF's hooked equivalent.
    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1])
        token_type_ids = torch.zeros_like(input_ids)
        x = model.token_embd(input_ids) + model.position_embd(position_ids)
        x = x + model.token_types(token_type_ids)
        x = model.token_embd_norm(x)
        for i, layer in enumerate(model.layers):
            x = layer(x)
            c = cosine(x, hf_hidden_states[f"layer_{i}"])
            ok = c >= COSINE_THRESHOLD
            all_ok &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] layer_{i}: cosine={c:.6f}")

    print("\nAll matched." if all_ok else "\nMismatches found - see FAIL lines above.")


if __name__ == "__main__":
    main()

"""Manual smoke test for M4: runs matricxon's own autoregressive loop
(KVCache + Sampler + ChatEngine) end to end against the real `ministral-3:3b`
weights and prints the result to eyeball for coherence.

Not a pass/fail check - ROADMAP.md's M4 calls for "multi-token generation
checked manually against the same model" specifically because "is this text
coherent" is a judgment call, not something to assert on (KVCache bounds and
Sampler determinism already have real unit tests). Run under a memory cap:

    systemd-run --user --scope -p MemoryMax=13G -- \\
        .venv/bin/python -m scripts.manual_generate_check

Unlike the M3 oracle scripts, this deliberately does NOT pass
`-p MemorySwapMax=0`: the full 26-layer model needs ~9GB+ resident even in
bf16 (see the note in `_load_model_in_bf16` below), which didn't fit this
machine's free RAM alone - it needed the swap headroom added specifically
for this. See scripts/README_oracle.md for the general memory-safety
approach this repo uses for model-loading work.
"""

import argparse

import torch

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.tokenizer import GGUFTokenizer
from scripts.oracle.common import DEFAULT_GGUF_PATH


def _load_model_in_bf16(gguf_path: str) -> Mistral3TextArchitecture:
    """Builds the *full* (all-layer) model directly in bf16 (via
    `from_gguf`'s explicit `dtype` parameter - see ModelManager, which loads
    real models the same way) rather than the float32 the production
    forward path defaults to - halves the ~13.7GB a real 3.85B-param
    float32 model would need, which doesn't fit this machine's 15GB RAM. M3
    already validated the forward math itself in float32; this script is
    only exercising the KVCache/Sampler/ChatEngine wiring on real weights.

    No `with` block around the loader (M10's on-the-fly dequant): real
    weight bytes aren't read at all until this model's first forward pass,
    so the loader has to stay open past this function returning - closed
    automatically by that first forward call instead (see
    ModelArchitecture._ensure_materialized).
    """
    loader = GGUFModelLoader(gguf_path, dtype=torch.bfloat16)
    model = Mistral3TextArchitecture.from_gguf(loader, dtype=torch.bfloat16)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", default=str(DEFAULT_GGUF_PATH))
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--num-predict", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = GGUFTokenizer(GGUFReader(args.gguf_path).read().metadata)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)

    model = _load_model_in_bf16(args.gguf_path)
    engine = ChatEngine(model, eos_token_ids={tokenizer.eos_token_id})

    sampling = SamplingConfig(
        temperature=args.temperature,
        num_ctx=input_ids.shape[1] + args.num_predict + 8,
        num_predict=args.num_predict,
        seed=0,
    )
    result = engine.generate(GenerationRequest(input_ids=input_ids, sampling=sampling))

    print(f"prompt:         {args.prompt!r}")
    print(f"finish_reason:  {result.finish_reason}")
    print(f"generated text: {tokenizer.decode(result.token_ids)!r}")


if __name__ == "__main__":
    main()

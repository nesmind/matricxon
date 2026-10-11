"""Manual smoke test (M4): runs matricxon's generation loop (KVCache + Sampler + ChatEngine) on the
real `ministral-3:3b` and prints the text to eyeball for coherence. Not pass/fail.

Run under a memory cap, without `-p MemorySwapMax=0`: the full model needs ~9GB+ even in bf16, so it
needs swap headroom (see tools/README_oracle.md):

    systemd-run --user --scope -p MemoryMax=13G -- \
        .venv/bin/python -m tools.manual_generate_check
"""

import argparse

import torch

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.tokenizer import GGUFTokenizer
from tools.oracle.common import DEFAULT_GGUF_PATH


def _load_model_in_bf16(gguf_path: str) -> Mistral3TextArchitecture:
    """Builds the full model in bf16 (via `from_gguf`'s `dtype`, as ModelManager), halving the
    ~13.7GB float32 needs. Only the KVCache/Sampler/ChatEngine wiring is exercised; M3 validated the
    forward math.

        No `with` block around the loader: weights are read at the first forward pass, which closes
        it (see ModelArchitecture._ensure_materialized).
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

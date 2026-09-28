"""Manual smoke test for Mixtral support in `LlamaArchitecture`: runs matricxon's own

autoregressive loop (KVCache + Sampler + ChatEngine) end to end against a real downloaded
Mixtral-architecture `llama` GGUF (e.g. `RichardErkhov/TitanML_-_tiny-mixtral-gguf`) and prints
the result to eyeball for coherence - same pattern as
`scripts/manual_generate_check_starcoder2.py`. This checkpoint is a real, tiny (2-layer,
untrained/toy-scale) Mixtral, so output is real code path proof (real sparse-MoE dispatch, real
router, real 3D expert tensors), not a coherence claim - a real production-scale Mixtral doesn't
fit this project's target hardware at all (smallest real one is 8x7B, ~47B total params).

    .venv/bin/python -m scripts.manual_generate_check_mixtral --gguf-path <path to .gguf>
"""

import argparse

import torch

from app.architectures.llama import LlamaArchitecture
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.models.tokenizer_dispatch import build_tokenizer
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--num-predict", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    metadata = GGUFReader(args.gguf_path).read().metadata
    print(f"architecture: {metadata.architecture}")
    print(f"expert_count: {metadata.get_u32('llama.expert_count')}")
    print(f"expert_used_count: {metadata.get_u32('llama.expert_used_count')}")

    tokenizer = build_tokenizer(metadata)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)

    loader = GGUFModelLoader(args.gguf_path, dtype=torch.float32)
    model = LlamaArchitecture.from_gguf(loader, dtype=torch.float32).eval()
    print(f"is_moe: {model.is_moe}, num_experts: {model.num_experts}")
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

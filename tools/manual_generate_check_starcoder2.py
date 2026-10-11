"""Manual smoke test for `starcoder2`: runs the generation loop (KVCache + Sampler + ChatEngine) on
a downloaded `bigcode/starcoder2-3b` GGUF and prints the text to eyeball, as
manual_generate_check.py (M4) but taking `--gguf-path`.

    .venv/bin/python -m tools.manual_generate_check_starcoder2 --gguf-path <path to .gguf>
"""

import argparse

import torch

from app.architectures.starcoder2 import Starcoder2Architecture
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.tokenizer import GGUFTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--prompt", default="def fibonacci(n):")
    parser.add_argument("--num-predict", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    metadata = GGUFReader(args.gguf_path).read().metadata
    print(f"architecture: {metadata.architecture}")
    print(f"tokenizer.ggml.model: {metadata.get_str('tokenizer.ggml.model')}")
    print(f"tokenizer.ggml.pre:   {metadata.get_str('tokenizer.ggml.pre')}")

    tokenizer = GGUFTokenizer(metadata)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)

    # bf16 halves the footprint vs float32 (as manual_generate_check.py).
    loader = GGUFModelLoader(args.gguf_path, dtype=torch.bfloat16)
    model = Starcoder2Architecture.from_gguf(loader, dtype=torch.bfloat16).eval()
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

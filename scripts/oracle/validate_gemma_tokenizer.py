"""Cross-checks Gemma4Tokenizer's encode() against the real HF tokenizer for

`google/gemma-4-12b-it` - the M10 gemma4-architecture equivalent of
scripts/oracle/validate_tokenizer.py (M5.5) and
validate_sentencepiece_tokenizer.py (M10 llama). Needs no memory cap: the
GGUF side only reads a KB-scale header, the HF side only a small
tokenizer.json fetch, never model weights.

    .venv/bin/python -m scripts.oracle.validate_gemma_tokenizer
"""

from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.runtime.gemma_tokenizer import Gemma4Tokenizer

GEMMA_GGUF_PATH = Path(
    "data/models/hf.co/google/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf"
)
GEMMA_HF_REPO = "google/gemma-4-12b-it"

_TEST_PROMPTS = [
    "The capital of France is",
    "hello world",
    "Hello, world! This is a test.",
    "Numbers: 12345 and 6.789, punctuation!! -- test.",
    "Emoji test: \U0001f389 café naïve résumé",
    "  leading spaces",
    "trailing spaces  ",
    "Repeated    spaces     test",
    "def foo(x):\n    return x + 1\n",
    "a",
    "",
    "<bos>What is 2+2?<eos>",
    "hello<bos>world",
]


def main() -> None:
    metadata = GGUFReader(GEMMA_GGUF_PATH).read().metadata
    ours = Gemma4Tokenizer(metadata)
    hf = AutoTokenizer.from_pretrained(GEMMA_HF_REPO)

    all_ok = True
    for prompt in _TEST_PROMPTS:
        our_ids = ours.encode(prompt)
        hf_ids = hf.encode(prompt, add_special_tokens=False)
        ok = our_ids == hf_ids
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}")
        if not ok:
            print(f"  ours: {our_ids}")
            print(f"  hf:   {hf_ids}")
            print(f"  ours pieces: {[ours._id_to_token[i] for i in our_ids]}")
            print(f"  hf pieces:   {hf.convert_ids_to_tokens(hf_ids)}")

    print(f"\n{'All prompts matched.' if all_ok else 'Mismatches found - see FAIL lines above.'}")


if __name__ == "__main__":
    main()

"""Cross-checks GGUFTokenizer's encode against the HF tokenizer for the same model (twin of
scripts/run_m3_oracle_check.sh for the forward pass). No memory cap: only a KB-scale GGUF header and
a small tokenizer.json.

Known, accepted mismatches (not bugs):
  - BOS: GGUF `add_bos_token=True` differs from the model's tokenizer_config.json
    (`add_bos_token=False`; the chat template adds BOS, which GGUFTokenizer leaves to the caller).
    Compared with `add_bos=False` here and `add_special_tokens=False` on HF.
  - Tabs: `tokenizer.ggml.pre="default"` (GPT-2-style regex) doesn't exactly match this
    tekken-derived tokenizer's whitespace handling around literal tabs. Rare in chat; not fixed.
"""

from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.runtime.tokenizer import GGUFTokenizer
from tools.oracle.common import DEFAULT_GGUF_PATH, HF_REPO

_TEST_PROMPTS = [
    "The capital of France is",
    "Hello, world! This is a test.",
    "Numbers: 12345 and 6.789, punctuation!! -- test.",
    "[INST] What is 2+2? [/INST]",
    "Emoji test: \U0001f389 café naïve résumé",
    "   leading and trailing spaces   ",
    "A longer paragraph with various punctuation: commas, periods. "
    'Question marks? Exclamation!! And "quotes" and (parentheses).',
    "Repeated    spaces     test",
]


def main() -> None:
    parsed = GGUFReader(Path(DEFAULT_GGUF_PATH)).read()
    ours = GGUFTokenizer(parsed.metadata)
    hf = AutoTokenizer.from_pretrained(HF_REPO)

    all_ok = True
    for prompt in _TEST_PROMPTS:
        our_ids = ours.encode(prompt)
        hf_ids = hf(prompt, add_special_tokens=False)["input_ids"]
        ok = our_ids == hf_ids
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}")
        if not ok:
            print(f"  ours: {our_ids}")
            print(f"  hf:   {hf_ids}")

    print(f"\n{'All prompts matched.' if all_ok else 'Mismatches found - see FAIL lines above.'}")


if __name__ == "__main__":
    main()

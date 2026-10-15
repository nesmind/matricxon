"""Cross-checks Matricxon's reranking (RerankEngine) against llama.cpp's rank pooling on a real
Qwen3-Reranker GGUF (e.g. ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF). Two steps, two venvs:

    .venv/bin/python -m tools.oracle.validate_qwen3_reranker mx MODEL.gguf out.json
    .venv-bench/bin/python tools/oracle/validate_qwen3_reranker.py llama MODEL.gguf out.json

The second step compares tokens and scores. This laptop overheats under load: run it on one core
with a temperature cutoff (see tools/thermal_guard.py)."""

import json
import sys

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower was finished in 1889.",
    "Bananas are rich in potassium.",
    "Berlin is the capital of Germany.",
]
TOLERANCE = 0.02


class MatricxonSide:
    def run(self, model_path: str, out_path: str) -> None:
        from types import SimpleNamespace

        from app.architectures.qwen3 import Qwen3Architecture
        from app.gguf.loader import GGUFModelLoader
        from app.models.tokenizer_dispatch import build_tokenizer
        from app.runtime.rerank_engine import DEFAULT_INSTRUCTION, Qwen3RerankPrompt, RerankEngine

        loader = GGUFModelLoader(model_path)
        architecture = Qwen3Architecture.from_gguf(loader)
        tokenizer = build_tokenizer(loader.metadata)
        engine = RerankEngine(
            SimpleNamespace(tag="oracle", architecture=architecture, tokenizer=tokenizer), 8192
        )
        prompts = [Qwen3RerankPrompt().build(QUERY, d, DEFAULT_INSTRUCTION) for d in DOCUMENTS]
        result = {
            "labels": architecture.score_labels,
            "scores": [engine.score(QUERY, d) for d in DOCUMENTS],
            "prompts": prompts,
            "tokens": [tokenizer.encode(p) for p in prompts],
        }
        with open(out_path, "w") as f:
            json.dump(result, f)
        print("matricxon:", [round(s, 4) for s in result["scores"]])


class LlamaSide:
    """With rank pooling llama.cpp returns the head's softmax, one probability per label
    (`[P(yes), P(no)]`), at the start of an n_embd-long vector (the rest is unused)."""

    def run(self, model_path: str, mx_path: str) -> None:
        import llama_cpp

        with open(mx_path) as f:
            mx = json.load(f)
        llm = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
            n_ctx=2048,
            n_threads=1,
            verbose=False,
        )
        ok = True
        for prompt, tokens, score in zip(mx["prompts"], mx["tokens"], mx["scores"], strict=True):
            same = llm.tokenize(prompt.encode(), add_bos=False, special=True) == tokens
            out = llm.embed(prompt, normalize=False)
            values = out[0] if out and isinstance(out[0], list) else out
            reference = values[mx["labels"].index("yes")]
            match = abs(reference - score) <= TOLERANCE
            ok = ok and same and match
            print(f"tokens_equal={same} llama={reference:.4f} mx={score:.4f}")
        print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    side, model, path = sys.argv[1:4]
    (MatricxonSide() if side == "mx" else LlamaSide()).run(model, path)

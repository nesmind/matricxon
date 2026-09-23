"""Manual benchmark: matricxon's own runtime vs llama-cpp-python on the *same* GGUF file, same
prompt, same settings - decides whether a llama.cpp backend is worth adding (see ROADMAP.md). Not a
pass/fail test, same as the other scripts/manual_*_check.py scripts.

Measures, per engine, client-side (matricxon's /api/chat has no separate prompt_eval_duration, so
both sides are timed identically from the outside):
- time to first token (TTFT) - dominated by prompt prefill on CPU;
- decode speed (tokens/s after the first token);
- a follow-up turn (turn 1's full history + a new question), which shows cross-request KV-cache
  reuse: llama-cpp-python keeps its cache between calls and only prefills the new suffix,
  matricxon re-prefills the whole history (see ROADMAP.md's "Cross-request KV-cache" entry).

The engines run one after the other, never together (this machine doesn't have the RAM for both):
matricxon's model is unloaded (keep_alive=0) before llama.cpp loads its own copy.

Setup - llama-cpp-python lives in its own venv, never matricxon's own, built from source with
GGML_NATIVE so it matches this CPU (a prebuilt wheel assumes AVX2, which this machine's i7-2640M
doesn't have):
    python3 -m venv .venv-bench
    CMAKE_ARGS="-DGGML_NATIVE=ON" .venv-bench/bin/pip install \\
        --no-binary llama-cpp-python llama-cpp-python

Usage (matricxon itself must already be running):
    .venv-bench/bin/python scripts/benchmark_llamacpp.py \\
        --tag hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M
"""

import argparse
import os
from pathlib import Path

from benchmark_runners import LlamaCppRunner, MatricxonRunner
from thermal_guard import ThermalGuard

DEFAULT_MODELS_DIR = os.environ.get(
    "MATRICXON_MODELS_DIR", "/home/home/Code/Py/AI/models/matricxon"
)
FIRST_QUESTION = (
    "Explain in a few paragraphs how a CPU cache works, including cache lines, the L1/L2/L3 "
    "hierarchy, and why cache misses are expensive."
)
FOLLOW_UP_QUESTION = "Now summarize that in three short bullet points."


def gguf_path_for_tag(models_dir: Path, tag: str) -> Path:
    """`hf.co/<org>/<repo>:<file>` -> `<models_dir>/hf.co/<org>/<repo>/<file>.gguf` - matricxon's
    own on-disk layout."""
    repo, filename = tag.split(":", 1)
    return models_dir / repo / f"{filename}.gguf"


def run_engine(
    name: str, runner: MatricxonRunner | LlamaCppRunner, repeats: int, guard: ThermalGuard
) -> dict:
    print(f"\n=== {name} ===")
    load_s = runner.load()
    print(f"load: {load_s:.1f}s")
    first_turn = [{"role": "user", "content": FIRST_QUESTION}]
    cold = []
    for i in range(repeats):
        if isinstance(runner, LlamaCppRunner):
            runner.reset_cache()  # a genuinely cold prompt, same as matricxon always gets
        guard.wait_until_cool()
        with guard.watch(runner.abort):
            result = runner.chat(first_turn)
        cold.append(result)
        print(
            f"turn 1, run {i + 1}: TTFT {result.ttft_s:.2f}s, {result.decode_tokens} tok in "
            f"{result.decode_s:.2f}s = {result.tokens_per_s:.2f} tok/s"
        )
    follow_up = [
        *first_turn,
        {"role": "assistant", "content": cold[-1].text},
        {"role": "user", "content": FOLLOW_UP_QUESTION},
    ]
    guard.wait_until_cool()
    # llama.cpp still holds turn 1 in its cache here - matricxon never does.
    with guard.watch(runner.abort):
        turn2 = runner.chat(follow_up)
    print(f"turn 2 (follow-up): TTFT {turn2.ttft_s:.2f}s, {turn2.tokens_per_s:.2f} tok/s")
    print(f"sample output: {cold[-1].text[:160]!r}")
    return {
        "load_s": load_s,
        "ttft_s": sum(r.ttft_s for r in cold) / len(cold),
        "tok_s": sum(r.tokens_per_s for r in cold) / len(cold),
        "turn2_ttft_s": turn2.ttft_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", required=True, help="matricxon model tag, e.g. hf.co/org/repo:file"
    )
    parser.add_argument(
        "--gguf", type=Path, help="GGUF path (default: derived from --tag and --models-dir)"
    )
    parser.add_argument("--models-dir", type=Path, default=Path(DEFAULT_MODELS_DIR))
    parser.add_argument("--host", default="http://localhost:8420", help="running matricxon server")
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count(),
        help="llama.cpp threads (matricxon's torch "
        "default is os.cpu_count(), so this matches it unless overridden)",
    )
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--skip-matricxon", action="store_true")
    parser.add_argument("--skip-llamacpp", action="store_true")
    parser.add_argument(
        "--max-temp", type=float, default=90.0, help="abort a run at this CPU temperature (C)"
    )
    parser.add_argument(
        "--resume-temp", type=float, default=72.0, help="wait for this CPU temperature (C) first"
    )
    args = parser.parse_args()
    guard = ThermalGuard(max_c=args.max_temp, resume_c=args.resume_temp)

    gguf_path = args.gguf or gguf_path_for_tag(args.models_dir, args.tag)
    if not gguf_path.is_file():
        parser.error(f"GGUF not found: {gguf_path} (pass --gguf explicitly)")
    options = {
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": 0.0,
        "seed": 42,
    }
    print(f"model: {gguf_path.name} ({gguf_path.stat().st_size / 1e9:.2f} GB), options: {options}")

    results = {}
    if not args.skip_matricxon:
        matricxon = MatricxonRunner(args.host, args.tag, options)
        results["matricxon"] = run_engine("matricxon", matricxon, args.repeats, guard)
        matricxon.unload()  # free its RAM before llama.cpp loads its own copy
    if not args.skip_llamacpp:
        results["llama.cpp"] = run_engine(
            "llama-cpp-python",
            LlamaCppRunner(gguf_path, options, args.threads),
            args.repeats,
            guard,
        )

    print("\n=== summary ===")
    print(f"{'engine':<12}{'load s':>9}{'TTFT s':>9}{'tok/s':>9}{'turn-2 TTFT s':>15}")
    for name, r in results.items():
        print(
            f"{name:<12}{r['load_s']:>9.1f}{r['ttft_s']:>9.2f}{r['tok_s']:>9.2f}{r['turn2_ttft_s']:>15.2f}"
        )
    if len(results) == 2:
        m, lc = results["matricxon"], results["llama.cpp"]
        print(
            f"\nllama.cpp vs matricxon: decode {lc['tok_s'] / max(m['tok_s'], 1e-9):.1f}x, "
            f"TTFT {m['ttft_s'] / max(lc['ttft_s'], 1e-9):.1f}x faster, "
            f"follow-up TTFT {m['turn2_ttft_s'] / max(lc['turn2_ttft_s'], 1e-9):.1f}x faster"
        )


if __name__ == "__main__":
    main()

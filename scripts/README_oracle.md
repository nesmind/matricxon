# M3 HF-oracle cross-check

Validates `Mistral3TextArchitecture`'s forward pass against a real Hugging
Face reference implementation, per [ROADMAP.md](../ROADMAP.md)'s M3 milestone.

## Run it

```bash
scripts/run_m3_oracle_check.sh
```

Override the model or layer count with `--gguf-path`, `--repo`, `--n-layers`
(forwarded to both dump scripts), or the memory cap with `ORACLE_MEMORY_MAX`
(default `4G`).

## Why it's built this way

This machine has 15GB RAM, no GPU, and swap is usually near-full from other
work - loading matricxon's own model *and* a second full-precision HF model
in the same process is what OOM-crashed VS Code the first time this was
attempted by hand. The design here avoids that on three levels:

- **Truncated depth.** Both sides only ever build the first `--n-layers`
  (default 4) decoder layers, never the full 26. The per-layer computation
  is identical at every depth, so this validates the same math while using
  a small fraction of the memory - see `common.DEFAULT_N_LAYERS`.
- **No bulk download.** The HF oracle side never downloads the 4.7GB
  `model.safetensors` file. `RemoteSafetensorsReader` (in `common.py`) reads
  the safetensors header via one HTTP range request, then fetches only the
  handful of tensors the truncated model actually needs, each via its own
  ranged GET - a few MB total instead of gigabytes. It's slow (dozens of
  serial round trips, mostly idle-CPU wait time) but memory- and
  bandwidth-cheap, which is the tradeoff that matters here.
- **Process isolation with a hard cap.** `dump_matricxon.py` and
  `dump_hf_oracle.py` each run as their own `systemd-run --user --scope`
  with `MemoryMax`/`MemorySwapMax=0`. If either one still overshoots, the
  kernel kills *that process* - not the desktop.

`compare.py` then loads only the small activation dumps (a few MB) and
needs no cap.

## Reading the output

Both dump scripts run in float32 throughout (matricxon's `nn.Linear`/
`nn.Embedding` parameters default to float32 and are never cast, regardless
of the loader's internal dequant dtype - see the comment in
`dump_matricxon.py`), so any mismatch reflects real numerical divergence,
not a bf16-vs-fp32 compute-precision artifact.

`compare.py` checks cosine similarity per tensor, not exact equality:
matricxon's weights are Q4_K/Q6_K (~4-6 bits/element) and the oracle's are
fp8/bf16 - both already lossy relative to the original checkpoint, so exact
equality isn't the bar. A `max_abs_diff` in the double digits on an
otherwise-passing tensor is expected, not a bug: transformer hidden states
routinely carry a few "massive activation" outlier dimensions (a
well-documented phenomenon), where quantization noise on that one dimension
produces a large absolute error while barely moving cosine similarity,
which is dominated by the many well-behaved dimensions.

A real bug looks very different from that noise floor: the RoPE-permutation
bug this check caught (see `_unpermute_rope_rows` in
`app/architectures/mistral3.py`) produced cosine similarities of 0.03-0.76 on
raw attention weights and forward activations alike - nowhere near the
~0.987-0.999 range normal quantization noise produces here.

# matricxon

A self-hosted, from-scratch Python + PyTorch inference runtime that implements
Ollama's HTTP API closely enough to be a drop-in replacement for
[pAIring]

Intended for ML students who want to learn how inference runtimes work
under the hood, and for private/personal use. **Not ready for production
use at the moment.**

No llama.cpp/ggml/vllm code or bindings are used: GGUF parsing, dequantization
kernels, the transformer forward pass, KV cache, and sampling are all
implemented here.

See [`ROADMAP.md`](ROADMAP.md) for the build order and current architecture
scope.

## Current status

matricxon now serves every endpoint pAIring actually calls: chat completion,
embeddings, and pulling models on its own, all backed by from-scratch GGUF
parsing, dequantization, transformer forward passes (both a causal decoder
and two non-causal encoder architectures), a KV cache, sampler, two
tokenizer implementations, a model manager with load/evict/keep-alive, and
a Hugging-Face-backed puller. Endpoints marked ⏳ below always return a
clean error rather than silently doing the wrong thing.

Pulling only ever resolves `hf.co/<repo>:<suffix>` tags, straight from
Hugging Face - not Ollama's registry, which matricxon has no access to
(that protocol is closed/undocumented). This isn't a compromise for any one
caller: it's the only tag shape matricxon can ever fetch real bytes for, for
anyone using it.

| Endpoint            | Status | Notes                                             |
| ------------------- | :----: | -------------------------------------------------- |
| `GET /api/tags`     |   ✅   | Lists installed models from the local catalog, including a real `estimated_ram_gb` per entry (exact per-tensor GGUF sizing, not an on-disk approximation) |
| `POST /api/show`    |   ✅   | Model details (family, params, context length, `estimated_ram_gb`) |
| `GET /api/ps`       |   ✅   | Reports actually-loaded models with live expiry    |
| `DELETE /api/delete`|   ✅   | Unloads first (if loaded), then removes from the catalog |
| `POST /api/chat`    |   ✅   | Real streaming NDJSON completion (an unload-only call returns `200` immediately); supports `tool_calls`/a `tool` role in the input message history and a top-level `tools` field |
| `POST /api/pull`    |   ✅   | Real download from `hf.co/<repo>:<suffix>` with progress + sha256 verification |
| `POST /api/embeddings` | ✅ | Real mean-pooled, L2-normalized embedding vector |
| `POST /api/generate` |  ✅   | Single raw-prompt completion (no chat template applied), same NDJSON/`ChatEngine` stack as `/api/chat` |
| `POST /api/embed`   |   ✅   | Batched sibling of `/api/embeddings` (`input: str \| list[str]`) - one `EmbeddingEngine` forward pass per item, not a real padded batch |
| `GET /api/version`  |   ✅   | Static version string |
| `POST /api/copy`    |   ✅   | Duplicates an installed model's blob (hardlinked where possible) + sidecar under a new tag |
| `POST /api/create`  |   ✅   | `FROM <existing-tag>`-only: a local re-tag via the same catalog duplication `/api/copy` uses. No Modelfile parser - `TEMPLATE`/`PARAMETER`/`SYSTEM` directives, if sent, are silently not applied |
| `POST /api/push`    |   ⏳   | Always fails closed (`501`) - matricxon has no registry to push to |
| `GET /api/health`   |   ✅   | Matricxon-only extension (not Ollama-compatible): real `supported_architectures`/`supported_quantizations` lists, read live off `ArchitectureRegistry`/`QuantStrategyRegistry` rather than hand-maintained |

Tool calling covers the *input* side only: a caller's message history can
include prior `tool_calls` (assistant turns) and `tool` role results, and
`Mistral3PromptBuilder` renders them into the real
`[AVAILABLE_TOOLS]`/`[TOOL_CALLS]`/`[ARGS]`/`[TOOL_RESULTS]` control-token
structure (confirmed against the real `chat_template.jinja` from
`mistralai/Ministral-3-3B-Instruct-2512`, not assumed). Parsing a
*generated* tool call back out of the model's own output into a structured
response `tool_calls` field isn't implemented yet - the raw
`[TOOL_CALLS]name[ARGS]{...}` text currently passes through as plain
`content`; a caller wanting to act on it must parse it itself for now.

`MATRICXON_MAX_LOADED_MODELS` (default `2`, matching pAIring's real
embed-then-chat usage pattern) can be raised on a machine with enough RAM -
`ModelManager`'s eviction/capacity logic is generic, not hardcoded to 2. A
load that clearly wouldn't fit now fails closed with a `503
InsufficientMemoryError` instead of risking an OS-level OOM.
`MATRICXON_MEMORY_SAFETY_MARGIN` (default `1.5`, bounded `1.1`-`1.8`) makes
that admission check's headroom requirement tunable per machine too,
instead of a hardcoded constant.

**Telemetry:** `MATRICXON_LOG_LEVEL` (default `0`, bounded `0`-`2`) turns on
stdlib `logging` output, previously absent everywhere in matricxon - level
`1` traces per-request pipeline stage boundaries (model load duration,
prompt build, each token's id/elapsed, generation summary); level `2` adds
a line per real decoder-layer iteration on every forward pass. Also
persisted via a `.env` file (real environment variables still take
priority) so a value set outside the current shell survives a restart.

**On-the-fly dequant:** `get_or_load()`/`from_gguf()` no longer copy any
real weight data in - they just build the (correctly-shaped, but
uninitialized) module graph and keep the GGUF file's mmap open. The actual
dequant happens once, lazily, on a model's first real forward pass, and is
cached for every call after that (see `ModelArchitecture._ensure_materialized`).
This means a request that fails validation (unknown model, prompt exceeds
`num_ctx`) no longer pays the multi-second-to-minutes dequant cost first -
it fails fast, before any real weight is ever touched - and a handle
evicted before ever being used pays no dequant cost at all. Total dequant
work for an actual generation is unchanged (same weights, same bytes),
just moved from load time to first-use time. Stopping a generation now also
takes effect mid-forward-pass, between decoder layers, rather than only
between whole generation calls - a `stop_check` threaded through every
architecture's forward pass raises cleanly instead of finishing a slow
prefill/materialization it no longer needs to.

**Quantized-native compute (opt-in):** set
`MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE=1` to skip full dequant on the
decode path. `QuantizedLinear` dispatches each single-token decode step
directly to a fused GEMV kernel running against the raw mmap'd quantized
bytes, covering all 11 packed GGUF quant types
(`Q2_K`-`Q8_K`/`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`); prefill is unaffected
and still dequantizes normally. Wired into `mistral3`, `llama`, `gemma4`,
and `phi2` (not `bert`/`nomic-bert`, which are always-prefill encoders).
Off by default - real measurement on this project's CPU-only target
hardware showed no speed benefit, but it's shipped as a real, permanent,
user-selectable choice rather than removed, since results may differ on
other hardware.

**GGUF architectures:** `mistral3` (text-only decoder; the vision tower is
parsed but its tensors are never read, per matricxon's v1
"accept-and-ignore images" scope), `bert` and `nomic-bert` (non-causal
encoders, embeddings only), `llama` (plain GQA decoder - RMSNorm/SwiGLU/
un-scaled RoPE, the "simpler special case" of `mistral3` without YaRN
scaling; validated against a real `TinyLlama-1.1B-Chat-v1.0` pull),
`gemma4` (dense, non-MoE - sandwich normalization, QK-norm, alternating
local/global attention layers with different head dims and RoPE bases,
final-logit softcapping; built from the real `transformers` Gemma4 source
against a real `google/gemma-4-12b-it` GGUF pull, but that model is too
large - ~28GB in bf16 - to run on this project's 15GB-RAM target hardware,
so only its wiring is validated against a tiny synthetic fixture, not its
numerics against real weights - a real, open gap, see ROADMAP.md), `phi2`
(parallel-residual decoder block - a single shared LayerNorm feeds both the
attention and MLP branches, unlike every other decoder here - partial
rotary embeddings, biases on every projection, and a fused `attn_qkv`
tensor split into query/key/value at materialize time; validated against a
real `moondream2-gguf` pull) - every other forward pass is cross-checked
against a real Hugging Face `transformers` model, see
[Testing](#testing) below. Any other `general.architecture` fails closed
with a clear error rather than attempting a best-effort forward pass.

**GGUF tokenizers:** byte-level BPE (`tokenizer.ggml.model = "gpt2"`),
WordPiece (`"bert"`), SentencePiece BPE (`"llama"` - score-based merge
selection and whole-text normalization, a genuinely different core
algorithm from the other two; cross-checked against the real HF
`LlamaTokenizer`, see `scripts/oracle/validate_sentencepiece_tokenizer.py`),
and Gemma4's own rank-based BPE (`"gemma4"` - like `"gpt2"`'s ordered-merge-
list algorithm, but no byte-to-unicode remapping and no regex pre-split
into words first; reverse-engineered against the real HF tokenizer's own
serialized `tokenizer.json`, cross-checked in
`scripts/oracle/validate_gemma_tokenizer.py`).

**Vision:** a standalone `ClipVisionEncoder` (`app/vision/`) implements
llama.cpp's real `clip`/mmproj GGUF format - a SigLIP-shaped ViT (patch
embedding, learned position embeddings, a pre-norm transformer stack) plus
a real MLP projector, with real image preprocessing
(`ClipImagePreprocessor`: base64 -> decode -> resize -> normalize with the
GGUF's own real `image_mean`/`image_std`). Validated against a real
`moondream/moondream2-gguf` mmproj pull (910MB) - correct output shape, no
NaN/Inf, and different real images produce genuinely different embeddings
(see `scripts/manual_vision_check.py`). Now wired into real end-to-end
`/api/chat` fusion: `app/runtime/vision_fusion.py` splits a prompt on
`[IMG]` markers, runs each image through `ClipVisionEncoder`, and splices
the resulting embeddings directly into the token-embedding tensor before
the decoder layers run, for the two architectures with a paired real
vision-language checkpoint - `llama` (LLaVA) and `phi2` (moondream2). The
paired mmproj file is found by convention (same directory,
`general.architecture = "clip"`) via `ModelCatalog.find_paired_mmproj()`,
and a model's `vision` capability is now computed dynamically from whether
that pairing exists, rather than fixed at pull time. Validated with real
streaming `/api/chat` generations against both real LLaVA and real
moondream2 GGUF files - coherent output, clean unload with no leaked mmap
references either time. Every other architecture (`mistral3`, `gemma4`,
`bert`, `nomic-bert`) still has no paired real vision-language checkpoint,
so their `ChatMessage.images` handling remains the pre-existing "count them
and insert `[IMG]` placeholder tokens, never read the actual bytes"
behavior.

`/api/chat`'s prompt building is currently hardcoded to
`Mistral3PromptBuilder` regardless of which architecture is actually
loaded - correct for `mistral3`, structurally wrong for `llama` (whose real
chat template uses `<|user|>`/`<|assistant|>`, not `[INST]`/`[/INST]`).
Tolerable for short completions (confirmed: still produces a correct
answer for a simple factual prompt) but a real multi-turn conversation
would get the wrong prompt structure - per-architecture prompt-builder
dispatch is real follow-up work, not yet implemented.

**GGUF quantization types:** `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_1`,
`Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`. Not yet
supported: the I-quants (`IQ*`, codebook/grid-based - a genuinely different
and larger undertaking than bit-unpacking) and ternary types - reading one of
these fails closed with `UnsupportedQuantTypeError` rather than
misinterpreting the bytes.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

## Running

matricxon defaults to port **8420**

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8420
```

Override host/port/models dir via env vars (see `app/config.py`), e.g.
`MATRICXON_PORT=12000`.

### Running as a background service

```bash
scripts/start.sh    # starts uvicorn detached, writes run/matricxon.pid + logs/matricxon.log
scripts/status.sh   # reports running/stopped + a liveness check against /api/tags
scripts/stop.sh     # graceful SIGTERM, falls back to SIGKILL after a timeout
```

All three respect `MATRICXON_HOST`/`MATRICXON_PORT` if set.

### Running as a systemd service (Debian/Ubuntu) 


## Testing

```bash
.venv/bin/pytest
.venv/bin/ruff check
```

### Validating a forward pass against a real model

Every architecture's forward pass, and both tokenizer implementations, are
cross-checked against a real Hugging Face `transformers` model or tokenizer -
not just unit-tested in isolation:

```bash
scripts/run_m3_oracle_check.sh                       # Mistral3TextArchitecture (truncated, memory-capped)
.venv/bin/python -m scripts.oracle.validate_tokenizer            # GGUFTokenizer (byte-level BPE)
.venv/bin/python -m scripts.oracle.validate_wordpiece_tokenizer  # WordPieceTokenizer
.venv/bin/python -m scripts.oracle.validate_bert                 # BertArchitecture
.venv/bin/python -m scripts.oracle.validate_nomic_bert           # NomicBertArchitecture
```

See [`scripts/README_oracle.md`](scripts/README_oracle.md) for how the
mistral3 check stays tractable on a machine with limited RAM and no GPU
(truncated layer count, no full-checkpoint download, memory-capped
subprocesses) - the encoder models are small enough (≤137M params) that
none of that is needed for `validate_bert`/`validate_nomic_bert`.

For a full end-to-end proof against real weights over the actual HTTP API
(not just the forward pass), see `scripts/manual_chat_check.py`,
`scripts/manual_embed_check.py`, and `scripts/manual_pull_check.py`.

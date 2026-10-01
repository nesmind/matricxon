<h1 align="center">
  <img src="assets/matricxon-logo.svg" alt="Matricxon - LLM inference runtime" width="460">
</h1>

A small, readable LLM inference runtime written from scratch in Python and PyTorch (C is used for the dequantization stages, for speed). It runs GGUF models locally - chat and text generation, embeddings, and vision-language
models - and serves them through an HTTP API, so tools like [pAIring](https://github.com/nesmind/pairing) can use all of its features.

The project started as part of the pAIring UI. We ran into limited advanced configuration options when working with Ollama, such as advanced memory management (for concurrent workloads) and others. We also made some important improvements to various inference stages, and everything is managed from the advanced pAIring UI, where Matricxon is the main inference engine.

For now, Matricxon is best suited to **learning and research**. It is written to be read, and it
shows how a GGUF file becomes tokens, how quantized weights are multiplied, how a KV cache and a
sampler work, and how hybrid models (attention plus state-space layers) are served.

Everything is built from scratch, with no llama.cpp, ggml or vLLM code or bindings: file parsing,
dequantization, the transformer forward passes, caching, sampling and scheduling. It runs on the CPUs only for now; we plan to add GPUs support soon.

## What's inside

| Part | What it does | Where |
| --- | --- | --- |
| GGUF reader | Parses metadata and tensors; memory-maps the file | `app/gguf/` |
| Dequantization | 25 quantization types, in Numba and native C | `app/gguf/dequant/`, `app/native/` |
| Architectures | Transformer, MoE, hybrid and encoder forward passes | `app/architectures/` |
| Runtime | Tokenizers, KV caches, sampler, generation loop, scheduler | `app/runtime/`, `app/models/` |
| Vision | Image encoders that feed vision-language models | `app/vision/` |
| Server | The HTTP API and the Hugging Face model puller | `app/routers/`, `app/pull/` |

## Supported models

| Architecture | Examples | Status |
| --- | --- | --- |
| `llama` | Llama 2/3, TinyLlama, Mixtral (MoE), LLaVA (vision) | checked on real weights |
| `mistral3` | Ministral | checked on real weights |
| `gemma4`, `granite`, `granitemoe`, `command-r`, `falcon` | Gemma 4 (dense and MoE), IBM Granite, Cohere Command R, Falcon | tiny-model tests only |
| `nemotron_h` | NVIDIA Nemotron-H (hybrid Mamba-2) | tiny-model tests only |
| `qwen2`, `qwen3` | Qwen 2.5, Qwen 3 | checked on real weights |
| `qwen35` | Qwen 3.5 (hybrid: Gated DeltaNet + attention), text and vision | checked on real weights |
| `phi2` | Phi-2, moondream2 (vision) | checked on real weights |
| `starcoder2` | StarCoder2 | checked on real weights |
| `bert`, `nomic-bert` | Embedding models | checked on real weights |

"Checked on real weights" means it was run on a real downloaded model and the output verified:
logits compared with Hugging Face `transformers` or llama.cpp, or coherent, correct generation.
"Tiny-model tests only" means the wiring is tested on small synthetic GGUF files but the numbers
are not yet verified on a full-size model. Any other architecture is rejected with a clear error,
never run approximately. Model families plug in through an architecture registry, so support for
more can be added without touching the rest.

**Quantization types:** `F32`, `F16`, `BF16`; `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`; the K-quants
`Q2_K` to `Q8_K`; and the I-quants and ternary types `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`,
`IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `TQ1_0`, `TQ2_0`. Not supported: `Q8_1` and the
raw integer types, which fail with a clear error.

**Tokenizers:** byte-level BPE, WordPiece, SentencePiece BPE and Gemma's rank-based BPE.

## Features

- **Quantized-native compute.** Weights stay packed in the memory-mapped file and are multiplied
  directly, with no full dequantized copy in RAM. A native C kernel (OpenMP threads) is the
  default; the Numba kernels are the fallback when there is no C compiler. On by default.
- **Lazy loading.** Loading a model only builds its structure; weight data is read on first use,
  so a bad request fails fast and an unused model costs no time.
- **Hybrid models.** Linear-attention and state-space layers keep a
  fixed-size recurrent state instead of a growing KV cache.
- **Prompt cache.** Each model keeps several conversations' caches, so users sharing a model don't
  make each other re-read their whole history. A new prompt reuses the cache that shares its
  longest prefix; if it doesn't continue that conversation, it copies just the shared part.
- **Concurrent users.** Replies on one model run side by side. A scheduler advances every active
  reply, and for most architectures their decode steps are batched into one forward pass.
  Batching helps most on memory-bandwidth-bound hardware; on a compute-bound CPU its main effect
  is that users stream at the same time instead of waiting in line. Mixture-of-experts models and
  Nemotron-H take one reply per step.
- **Vision.** CLIP/SigLIP-style projector files feed image embeddings into `llama` (LLaVA),
  `phi2` (moondream2) and `qwen35`. The projector is found next to the model by convention.
- **Safe memory use.** A model that would not fit in free RAM (with a safety margin) is refused
  instead of risking the OS killing the process. Idle models unload after a keep-alive time.
- **Cancelling one reply.** A `request_id` on `/api/chat` stops just that reply; the model and
  other users' replies are untouched.

## API

| Endpoint | Notes |
| --- | --- |
| `POST /api/chat` | Streaming NDJSON chat with tools and image support |
| `POST /api/generate` | Raw-prompt completion, no chat template |
| `POST /api/embeddings`, `POST /api/embed` | Mean-pooled, L2-normalized vectors (single and list input) |
| `POST /api/pull` | Downloads `hf.co/<repo>:<file>` from Hugging Face with progress and sha256 check |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps` | Installed models, model details, loaded models with expiry |
| `DELETE /api/delete`, `POST /api/copy`, `POST /api/create` | Remove, duplicate, re-tag (`create` supports only `FROM <tag>`) |
| `GET /api/health` | Matricxon-only: supported architectures, quantizations, MoE and vision lists |
| `GET /api/version` | Version string |

Pulling resolves only `hf.co/<repo>:<file>` tags; Ollama's own registry is not reachable.
`POST /api/push` always returns `501`. A generated `tool_calls` field is not parsed back out of
the model's text yet.

## Setup

```bash
git clone https://github.com/nesmind/matricxon.git
cd matricxon
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # requirements-dev.txt adds the test tools
```

The native C kernels are compiled automatically on first start if `gcc` (or `cc` on macOS) is
available. Without a compiler, Matricxon falls back to the Numba kernels, which are much slower.

**macOS:** `scripts/install_mac.sh` checks the prerequisites, optionally installs `libomp` (for
multi-threaded kernels), creates `.venv`, installs the dependencies, writes a starter `.env` and
builds the native kernels once so you can see right away whether they built:

```bash
scripts/install_mac.sh          # add --yes to install libomp without asking
```

## Running

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8420
```

Or in the background, with a PID file and a log:

```bash
scripts/start.sh     # starts detached; writes run/matricxon.pid and logs/matricxon.log
scripts/status.sh    # running or stopped, plus a liveness check
scripts/stop.sh      # graceful stop
```

## Configuration

Set environment variables, or put them in a `.env` file in the project folder (real environment
variables win). All of them are listed in `app/config.py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MATRICXON_HOST`, `MATRICXON_PORT` | `0.0.0.0`, `8420` | Where the server listens |
| `MATRICXON_MODELS_DIR` | `./data/models` | Where models are stored |
| `MATRICXON_MAX_LOADED_MODELS` | `2` | Models kept in RAM at once |
| `MATRICXON_DEFAULT_KEEP_ALIVE_SECONDS` | `300` | Idle time before a model unloads |
| `MATRICXON_MEMORY_SAFETY_MARGIN` | `1.2` | Free RAM needed beyond a model's size (1.1 to 1.8) |
| `MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE` | `true` | Multiply packed weights directly |
| `MATRICXON_GEMV_BACKEND` | `native` | `native` (C kernels) or `numba` |
| `MATRICXON_TORCH_THREADS` | all cores | CPU threads; fewer runs cooler |
| `MATRICXON_MAX_DECODE_BATCH` | `8` | Replies that run at once per model |
| `MATRICXON_PROMPT_CACHE_SLOTS` | `4` | Conversations cached per model |
| `MATRICXON_PROMPT_CACHE_BUDGET_MB` | `2048` | Memory those caches may use together |
| `MATRICXON_LOG_LEVEL` | `0` | `0` warnings, `1` per-request steps, `2` per-layer trace |

## Testing

```bash
.venv/bin/pytest            # unit and API tests (tiny synthetic models, no downloads)
.venv/bin/ruff check
```

Forward passes and tokenizers are also compared with Hugging Face on real models:

```bash
.venv/bin/python -m scripts.oracle.validate_qwen3          # also validate_qwen2, validate_bert, ...
.venv/bin/python -m scripts.oracle.validate_tokenizer
```

See [`scripts/README_oracle.md`](scripts/README_oracle.md) for running these on a machine with
limited RAM. `scripts/manual_chat_check.py`, `manual_embed_check.py` and `manual_pull_check.py`
exercise the real HTTP API end to end. Long benchmarks can overheat a laptop: use
`scripts/thermal_guard.py` to stop them at a safe temperature.

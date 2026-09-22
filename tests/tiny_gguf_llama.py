"""Builds a tiny, complete, valid `llama` GGUF file - real architecture,

real SentencePiece tokenizer metadata, all F32 - so integration tests can
run the *entire* real pipeline (GGUFReader -> GGUFModelLoader ->
ArchitectureRegistry -> LlamaArchitecture -> SentencePieceTokenizer ->
ChatEngine -> NDJSON) end to end, fast and memory-safely. The generated
text is gibberish (tiny near-random weights) - this is for exercising the
wiring, not model quality; the real TinyLlama-1.1B-Chat-v1.0 pull already
validated the real math (see ROADMAP.md's M10 llama-architecture entry, and
scripts/oracle/validate_sentencepiece_tokenizer.py for the tokenizer side).

Vocab is deliberately minimal, same philosophy as tiny_gguf.py's GPT-2
fixture: control tokens (type 3) plus all 256 byte-fallback tokens (type 6,
`<0xNN>`) and nothing else - every character falls through to byte-fallback,
so no real BPE merge ever needs to succeed (`tokenizer.ggml.scores` can all
be 0.0 unused) while `encode`/`decode` still round-trip correctly for any
input text.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 4
N_LAYER = 1
FFN_LEN = 16

_CONTROL_TOKENS = ["<unk>", "<s>", "</s>"]
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def _byte_tokens() -> list[str]:
    return [f"<0x{b:02X}>" for b in range(256)]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def build_tiny_llama_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [6] * 256
    scores = [0.0] * len(tokens)
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "llama")
        .set_u32("llama.embedding_length", N_EMBD)
        .set_u32("llama.attention.head_count", N_HEAD)
        .set_u32("llama.attention.head_count_kv", N_HEAD_KV)
        .set_u32("llama.block_count", N_LAYER)
        .set_u32("llama.feed_forward_length", FFN_LEN)
        .set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("llama.vocab_size", vocab_size)
        .set_f32("llama.rope.freq_base", 10000.0)
        .set_str("tokenizer.ggml.model", "llama")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, scores)
        .set_array("tokenizer.ggml.token_type", GGUFValueType.INT32, token_types)
        .set_u32("tokenizer.ggml.bos_token_id", BOS_TOKEN_ID)
        .set_u32("tokenizer.ggml.eos_token_id", EOS_TOKEN_ID)
    )

    def add(name: str, array: np.ndarray) -> None:
        # GGUF's ne[] is fastest-dim-first (PyTorch's shape reversed) - see
        # the row-permute note in app/architectures/layers.py.
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add("token_embd.weight", _random_weight(rng, vocab_size, N_EMBD))
    # Real Llama-3.2-1B/3B-Instruct GGUFs have no separate "output.weight" tensor at all
    # (tie_word_embeddings=true) - see LlamaArchitecture's own docstring for the real bug this
    # covers (2026-09-22).
    if not tied_embeddings:
        add("output.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add("blk.0.attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add("blk.0.ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add("blk.0.attn_q.weight", _random_weight(rng, N_HEAD * HEAD_DIM, N_EMBD))
    add("blk.0.attn_k.weight", _random_weight(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
    add("blk.0.attn_v.weight", _random_weight(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
    add("blk.0.attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * HEAD_DIM))
    add("blk.0.ffn_gate.weight", _random_weight(rng, FFN_LEN, N_EMBD))
    add("blk.0.ffn_up.weight", _random_weight(rng, FFN_LEN, N_EMBD))
    add("blk.0.ffn_down.weight", _random_weight(rng, N_EMBD, FFN_LEN))

    return builder.write(path)

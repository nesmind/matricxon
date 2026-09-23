"""A tiny `llama` GGUF whose 2-D weights are Q8_0-quantized (norms stay F32), with tied embeddings
like a real Llama-3.2 file - big enough (n_embd 64) for Q8_0's 32-value blocks, so the packed
paths (QuantizedEmbedding, a tied QuantizedLinear lm_head, attn_q/attn_k with their row
permutation applied to packed rows) can be checked end to end against the float path on the
exact same quantized values. Tokenizer metadata matches tests/tiny_gguf_llama.py.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder
from tests.tiny_gguf_llama import _CONTROL_TOKENS, _byte_tokens

N_EMBD = 64
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16
FFN_LEN = 128


def _q8_0_bytes(array: np.ndarray) -> bytes:
    """Q8_0: per 32 values, {f16 d; i8 qs[32]} with d = amax/127."""
    blocks = array.astype(np.float32).reshape(-1, 32)
    d = np.abs(blocks).max(axis=1, keepdims=True) / 127.0
    d[d == 0] = 1.0
    qs = np.clip(np.rint(blocks / d), -127, 127).astype(np.int8)
    out = bytearray()
    for scale, row in zip(d[:, 0].astype(np.float16), qs, strict=True):
        out += scale.tobytes() + row.tobytes()
    return bytes(out)


def build_tiny_llama_q8_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [6] * 256
    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "llama")
        .set_u32("llama.embedding_length", N_EMBD)
        .set_u32("llama.attention.head_count", N_HEAD)
        .set_u32("llama.attention.head_count_kv", N_HEAD_KV)
        .set_u32("llama.block_count", 1)
        .set_u32("llama.feed_forward_length", FFN_LEN)
        .set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("llama.vocab_size", len(tokens))
        .set_f32("llama.rope.freq_base", 10000.0)
        .set_str("tokenizer.ggml.model", "llama")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, [0.0] * len(tokens))
        .set_array("tokenizer.ggml.token_type", GGUFValueType.INT32, token_types)
        .set_u32("tokenizer.ggml.bos_token_id", 1)
        .set_u32("tokenizer.ggml.eos_token_id", 2)
    )

    def add_q8(name: str, out_features: int, in_features: int) -> None:
        weight = rng.randn(out_features, in_features) * 0.1
        builder.add_tensor(
            name, [in_features, out_features], GGMLQuantizationType.Q8_0, _q8_0_bytes(weight)
        )

    def add_norm(name: str) -> None:
        builder.add_tensor(
            name, [N_EMBD], GGMLQuantizationType.F32, np.ones(N_EMBD, dtype="<f4").tobytes()
        )

    add_q8("token_embd.weight", len(tokens), N_EMBD)
    add_norm("output_norm.weight")
    add_norm("blk.0.attn_norm.weight")
    add_norm("blk.0.ffn_norm.weight")
    add_q8("blk.0.attn_q.weight", N_HEAD * HEAD_DIM, N_EMBD)
    add_q8("blk.0.attn_k.weight", N_HEAD_KV * HEAD_DIM, N_EMBD)
    add_q8("blk.0.attn_v.weight", N_HEAD_KV * HEAD_DIM, N_EMBD)
    add_q8("blk.0.attn_output.weight", N_EMBD, N_HEAD * HEAD_DIM)
    add_q8("blk.0.ffn_gate.weight", FFN_LEN, N_EMBD)
    add_q8("blk.0.ffn_up.weight", FFN_LEN, N_EMBD)
    add_q8("blk.0.ffn_down.weight", N_EMBD, FFN_LEN)
    return builder.write(path)

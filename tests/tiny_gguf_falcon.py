"""Builds a tiny, complete, valid `falcon` GGUF file - mirrors `tiny_gguf_command_r.py`'s

structure, but with a real fused, bias-free `attn_qkv` tensor in real MQA proportions
(`head_count=2`, `head_count_kv=1` here - real `tiiuae/falcon-7b` is `71`/`1`) instead of
separate `attn_q`/`attn_k`/`attn_v`, and a real **biased** `attn_norm` (the one shared
parallel-residual norm - no `attn_norm_2` tensor at all, confirmed real for this
`parallel_attn=True, new_decoder_architecture=False` variant) feeding a plain, bias-free,
non-gated GELU MLP (`ffn_up`/`ffn_down`, no `ffn_gate`).
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
    from app.runtime.tokenizer import _byte_to_unicode

    byte_encoder = _byte_to_unicode()
    return [byte_encoder[b] for b in range(256)]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def _random_norm_weight(rng: np.random.RandomState, dim: int) -> np.ndarray:
    return (np.ones(dim) + rng.randn(dim) * 0.05).astype("<f4")


def _random_bias(rng: np.random.RandomState, dim: int) -> np.ndarray:
    return (rng.randn(dim) * 0.02).astype("<f4")


def build_tiny_falcon_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [1] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "falcon")
        .set_u32("falcon.embedding_length", N_EMBD)
        .set_u32("falcon.attention.head_count", N_HEAD)
        .set_u32("falcon.attention.head_count_kv", N_HEAD_KV)
        .set_u32("falcon.block_count", N_LAYER)
        .set_u32("falcon.feed_forward_length", FFN_LEN)
        .set_f32("falcon.attention.layer_norm_epsilon", 1e-5)
        .set_u32("falcon.vocab_size", vocab_size)
        .set_str("tokenizer.ggml.model", "gpt2")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.merges", GGUFValueType.STRING, [])
        .set_array("tokenizer.ggml.token_type", GGUFValueType.INT32, token_types)
        .set_u32("tokenizer.ggml.bos_token_id", BOS_TOKEN_ID)
        .set_u32("tokenizer.ggml.eos_token_id", EOS_TOKEN_ID)
    )

    def add(name: str, array: np.ndarray) -> None:
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add("token_embd.weight", _random_weight(rng, vocab_size, N_EMBD))
    if not tied_embeddings:
        add("output.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("output_norm.weight", _random_norm_weight(rng, N_EMBD))
    add("output_norm.bias", _random_bias(rng, N_EMBD))

    add("blk.0.attn_norm.weight", _random_norm_weight(rng, N_EMBD))
    add("blk.0.attn_norm.bias", _random_bias(rng, N_EMBD))
    qkv_rows = N_HEAD * HEAD_DIM + 2 * N_HEAD_KV * HEAD_DIM
    add("blk.0.attn_qkv.weight", _random_weight(rng, qkv_rows, N_EMBD))
    add("blk.0.attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * HEAD_DIM))

    add("blk.0.ffn_up.weight", _random_weight(rng, FFN_LEN, N_EMBD))
    add("blk.0.ffn_down.weight", _random_weight(rng, N_EMBD, FFN_LEN))

    return builder.write(path)

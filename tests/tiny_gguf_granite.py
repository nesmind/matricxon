"""Builds a tiny, complete, valid `granite` GGUF file - mirrors `tiny_gguf_llama.py`'s exact
structure/philosophy, plus IBM Granite's four real "Power"-scaling metadata keys
(`granite.attention.scale`/`embedding_scale`/`residual_scale`/`logit_scale`) set to deliberately
non-default (non-1.0) values, so a bug that silently no-ops any one multiplier changes the
numeric forward-pass output, not just shapes - this proves the wiring is real, not that it
numerically matches a real Granite checkpoint (see `scripts/oracle/validate_granite.py` for that).
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

ATTENTION_SCALE = 0.5
EMBEDDING_MULTIPLIER = 2.0
RESIDUAL_MULTIPLIER = 0.5
LOGITS_SCALING = 2.0

_CONTROL_TOKENS = ["<unk>", "<s>", "</s>"]
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def _byte_tokens() -> list[str]:
    from app.runtime.tokenizer import _byte_to_unicode

    byte_encoder = _byte_to_unicode()
    return [byte_encoder[b] for b in range(256)]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def build_tiny_granite_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [1] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "granite")
        .set_u32("granite.embedding_length", N_EMBD)
        .set_u32("granite.attention.head_count", N_HEAD)
        .set_u32("granite.attention.head_count_kv", N_HEAD_KV)
        .set_u32("granite.block_count", N_LAYER)
        .set_u32("granite.feed_forward_length", FFN_LEN)
        .set_f32("granite.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("granite.vocab_size", vocab_size)
        .set_f32("granite.rope.freq_base", 10000.0)
        .set_f32("granite.attention.scale", ATTENTION_SCALE)
        .set_f32("granite.embedding_scale", EMBEDDING_MULTIPLIER)
        .set_f32("granite.residual_scale", RESIDUAL_MULTIPLIER)
        .set_f32("granite.logit_scale", LOGITS_SCALING)
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

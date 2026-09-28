"""Builds a tiny, complete, valid Mixtral-style `llama` GGUF file - mirrors `tiny_gguf_llama.py`'s

structure plus real MoE metadata (`llama.expert_count`/`expert_used_count`) and the real
llama.cpp Mixtral tensor convention (`ffn_gate_inp`/`ffn_gate_exps`/`ffn_up_exps`/
`ffn_down_exps`, one 3D tensor per projection type) instead of plain `ffn_gate`/`ffn_up`/
`ffn_down` - confirmed real (2026-09-30): llama.cpp has no separate `mixtral` architecture string
at all, `general.architecture` stays `"llama"` for a real Mixtral pull too (see
`app/architectures/llama_moe.py`'s own module docstring). `N_EXPERTS=4`/`EXPERTS_PER_TOK=2` with
a multi-token prompt in tests exercises both "not every expert selected" and "multiple tokens
pick overlapping experts" real code paths, same reasoning `tiny_gguf_granitemoe.py` already
established.
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
N_EXPERTS = 4
EXPERTS_PER_TOK = 2

_CONTROL_TOKENS = ["<unk>", "<s>", "</s>"]
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def _byte_tokens() -> list[str]:
    return [f"<0x{b:02X}>" for b in range(256)]


def _random_weight(rng: np.random.RandomState, *shape: int) -> np.ndarray:
    return (rng.randn(*shape) * 0.02).astype("<f4")


def build_tiny_llama_moe_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
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
        .set_u32("llama.expert_count", N_EXPERTS)
        .set_u32("llama.expert_used_count", EXPERTS_PER_TOK)
        .set_str("tokenizer.ggml.model", "llama")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, scores)
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
    add("blk.0.ffn_gate_inp.weight", _random_weight(rng, N_EXPERTS, N_EMBD))
    add("blk.0.ffn_gate_exps.weight", _random_weight(rng, N_EXPERTS, FFN_LEN, N_EMBD))
    add("blk.0.ffn_up_exps.weight", _random_weight(rng, N_EXPERTS, FFN_LEN, N_EMBD))
    add("blk.0.ffn_down_exps.weight", _random_weight(rng, N_EXPERTS, N_EMBD, FFN_LEN))

    return builder.write(path)

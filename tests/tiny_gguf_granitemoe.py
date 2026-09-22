"""Builds a tiny, complete, valid `granitemoe` GGUF file - mirrors `tiny_gguf_granite.py`'s
structure plus real MoE metadata (`granitemoe.expert_count`/`expert_used_count`) and the real
llama.cpp Mixtral-style per-expert 3D tensor convention (`ffn_gate_inp`/`ffn_gate_exps`/
`ffn_up_exps`/`ffn_down_exps`, one 3D tensor per projection type rather than N separate 2D
tensors). `N_EXPERTS=4`/`EXPERTS_PER_TOK=2` with a multi-token prompt in tests exercises both
"not every expert selected" and "multiple tokens pick overlapping experts" real code paths.
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


def _random_weight(rng: np.random.RandomState, *shape: int) -> np.ndarray:
    return (rng.randn(*shape) * 0.02).astype("<f4")


def build_tiny_granitemoe_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [1] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "granitemoe")
        .set_u32("granitemoe.embedding_length", N_EMBD)
        .set_u32("granitemoe.attention.head_count", N_HEAD)
        .set_u32("granitemoe.attention.head_count_kv", N_HEAD_KV)
        .set_u32("granitemoe.block_count", N_LAYER)
        .set_u32("granitemoe.feed_forward_length", FFN_LEN)
        .set_f32("granitemoe.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("granitemoe.vocab_size", vocab_size)
        .set_f32("granitemoe.rope.freq_base", 10000.0)
        .set_u32("granitemoe.expert_count", N_EXPERTS)
        .set_u32("granitemoe.expert_used_count", EXPERTS_PER_TOK)
        .set_f32("granitemoe.attention.scale", ATTENTION_SCALE)
        .set_f32("granitemoe.embedding_scale", EMBEDDING_MULTIPLIER)
        .set_f32("granitemoe.residual_scale", RESIDUAL_MULTIPLIER)
        .set_f32("granitemoe.logit_scale", LOGITS_SCALING)
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
    add("blk.0.ffn_gate_inp.weight", _random_weight(rng, N_EXPERTS, N_EMBD))
    add("blk.0.ffn_gate_exps.weight", _random_weight(rng, N_EXPERTS, FFN_LEN, N_EMBD))
    add("blk.0.ffn_up_exps.weight", _random_weight(rng, N_EXPERTS, FFN_LEN, N_EMBD))
    add("blk.0.ffn_down_exps.weight", _random_weight(rng, N_EXPERTS, N_EMBD, FFN_LEN))

    return builder.write(path)

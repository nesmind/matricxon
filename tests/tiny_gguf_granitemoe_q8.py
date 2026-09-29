"""A tiny `granitemoe` GGUF whose expert tensors (`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`)
are Q8_0-quantized - mirrors `tiny_gguf_granitemoe.py`'s structure with bigger, block-aligned
dims (n_embd/ffn_len both 32, Q8_0's own block size), so the real `QuantizedMoEExperts` packed
path (`app/architectures/moe_experts.py`) can be checked end to end against the dense/F32 path
on the exact same quantized values - same precedent as `tests/tiny_gguf_llama_q8.py`. The router
(`ffn_gate_inp`) and attention tensors stay F32 - neither is ever packed (see
`QuantizedMoEExperts`'s own docstring for the router; attention here is small and not the
concern of this fixture).
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder
from tests.tiny_gguf_llama_q8 import _q8_0_bytes

# 256 (not just a Q8_0-block-aligned 32) so the native C backend's own real block-size
# requirement for this type (MX_QK_K=256, see mx_common.h) is satisfied too - a smaller size
# would silently only ever exercise the Numba/dequant fallback, never native C.
N_EMBD = 256
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 128  # N_HEAD * HEAD_DIM == N_EMBD, same real relationship every actual checkpoint has
N_LAYER = 1
FFN_LEN = 256
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


def _random_f32(rng: np.random.RandomState, *shape: int) -> np.ndarray:
    return (rng.randn(*shape) * 0.02).astype("<f4")


def build_tiny_granitemoe_q8_gguf(path: Path, seed: int = 0) -> Path:
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

    def add_f32(name: str, array: np.ndarray) -> None:
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    def add_q8_3d(name: str, num_experts: int, out_features: int, in_features: int) -> None:
        weight = rng.randn(num_experts, out_features, in_features) * 0.1
        builder.add_tensor(
            name,
            [in_features, out_features, num_experts],
            GGMLQuantizationType.Q8_0,
            _q8_0_bytes(weight),
        )

    add_f32("token_embd.weight", _random_f32(rng, vocab_size, N_EMBD))
    add_f32("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32("blk.0.attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32("blk.0.ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32("blk.0.attn_q.weight", _random_f32(rng, N_HEAD * HEAD_DIM, N_EMBD))
    add_f32("blk.0.attn_k.weight", _random_f32(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
    add_f32("blk.0.attn_v.weight", _random_f32(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
    add_f32("blk.0.attn_output.weight", _random_f32(rng, N_EMBD, N_HEAD * HEAD_DIM))
    add_f32("blk.0.ffn_gate_inp.weight", _random_f32(rng, N_EXPERTS, N_EMBD))
    add_q8_3d("blk.0.ffn_gate_exps.weight", N_EXPERTS, FFN_LEN, N_EMBD)
    add_q8_3d("blk.0.ffn_up_exps.weight", N_EXPERTS, FFN_LEN, N_EMBD)
    add_q8_3d("blk.0.ffn_down_exps.weight", N_EXPERTS, N_EMBD, FFN_LEN)

    return builder.write(path)

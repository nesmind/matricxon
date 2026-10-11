"""Builds a tiny, complete, valid `gemma4` GGUF with the architecture's shape (local/sliding and
global layer types, different head_dim per type, the `rope_freqs.weight` correction tensor and
final-logit softcapping), minimal Gemma4Tokenizer-shaped metadata, all F32, so integration tests run
the *entire* pipeline (GGUFReader -> GGUFModelLoader -> Gemma4Architecture -> Gemma4Tokenizer ->
ChatEngine -> NDJSON) fast and memory-safely.

It proves the *wiring* (shapes, per-layer-type dispatch, sliding-window masking, v-from-k reuse on
the global layer, the rope_freqs path, sandwich-norm ordering) doesn't crash and gives correctly
shaped output, not a numeric match to `google/gemma-4-12b-it`, whose 7GB GGUF is impractical to
validate on 15GB-RAM hardware even in bf16 (~28GB estimated); ROADMAP.md M10 gemma4: accepted gap.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from tools.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
N_LAYER = 6
FFN_LEN = 16
SLIDING_WINDOW = 4
HEAD_DIM_LOCAL = 4
HEAD_DIM_GLOBAL = 6
N_HEAD_KV_LOCAL = 2
N_HEAD_KV_GLOBAL = 1
FINAL_LOGIT_SOFTCAPPING = 30.0

_CONTROL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 1


def _is_sliding(layer_idx: int) -> bool:
    return layer_idx % 6 != 5


def _byte_tokens() -> list[str]:
    return [f"<0x{b:02X}>" for b in range(256)]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def build_tiny_gemma4_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [6] * 256
    scores = [0.0] * len(tokens)
    vocab_size = len(tokens)

    head_count_kv = [
        N_HEAD_KV_LOCAL if _is_sliding(i) else N_HEAD_KV_GLOBAL for i in range(N_LAYER)
    ]
    sliding_pattern = [_is_sliding(i) for i in range(N_LAYER)]

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "gemma4")
        .set_u32("gemma4.embedding_length", N_EMBD)
        .set_u32("gemma4.attention.head_count", N_HEAD)
        .set_array("gemma4.attention.head_count_kv", GGUFValueType.UINT32, head_count_kv)
        .set_array("gemma4.attention.sliding_window_pattern", GGUFValueType.BOOL, sliding_pattern)
        .set_u32("gemma4.attention.sliding_window", SLIDING_WINDOW)
        .set_u32("gemma4.attention.key_length", HEAD_DIM_GLOBAL)
        .set_u32("gemma4.attention.key_length_swa", HEAD_DIM_LOCAL)
        .set_u32("gemma4.block_count", N_LAYER)
        .set_u32("gemma4.feed_forward_length", FFN_LEN)
        .set_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
        .set_f32("gemma4.rope.freq_base", 1000000.0)
        .set_f32("gemma4.rope.freq_base_swa", 10000.0)
        .set_f32("gemma4.final_logit_softcapping", FINAL_LOGIT_SOFTCAPPING)
        .set_str("tokenizer.ggml.model", "gemma4")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, scores)
        .set_array("tokenizer.ggml.merges", GGUFValueType.STRING, [])
        .set_array("tokenizer.ggml.token_type", GGUFValueType.INT32, token_types)
        .set_u32("tokenizer.ggml.bos_token_id", BOS_TOKEN_ID)
        .set_u32("tokenizer.ggml.eos_token_id", EOS_TOKEN_ID)
    )

    def add(name: str, array: np.ndarray) -> None:
        # GGUF's ne[] is fastest-dim-first (PyTorch's shape reversed); see the row-permute note in
        # app/architectures/layers.py.
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add("token_embd.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    # llama.cpp's freq_factors convention: a no-op correction (all 1.0) exercises the code path
    # without a faked meaningful value.
    add("rope_freqs.weight", np.ones(HEAD_DIM_GLOBAL // 2, dtype="<f4"))

    for i in range(N_LAYER):
        is_sliding = _is_sliding(i)
        head_dim = HEAD_DIM_LOCAL if is_sliding else HEAD_DIM_GLOBAL
        n_head_kv = N_HEAD_KV_LOCAL if is_sliding else N_HEAD_KV_GLOBAL
        prefix = f"blk.{i}."

        add(prefix + "attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "attn_q.weight", _random_weight(rng, N_HEAD * head_dim, N_EMBD))
        add(prefix + "attn_q_norm.weight", np.ones(head_dim, dtype="<f4"))
        add(prefix + "attn_k.weight", _random_weight(rng, n_head_kv * head_dim, N_EMBD))
        add(prefix + "attn_k_norm.weight", np.ones(head_dim, dtype="<f4"))
        if is_sliding:
            add(prefix + "attn_v.weight", _random_weight(rng, n_head_kv * head_dim, N_EMBD))
        add(prefix + "attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * head_dim))
        add(prefix + "post_attention_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_gate.weight", _random_weight(rng, FFN_LEN, N_EMBD))
        add(prefix + "ffn_up.weight", _random_weight(rng, FFN_LEN, N_EMBD))
        add(prefix + "ffn_down.weight", _random_weight(rng, N_EMBD, FFN_LEN))
        add(prefix + "post_ffw_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "layer_output_scale.weight", np.ones(1, dtype="<f4"))

    return builder.write(path)


# Per-Layer Embeddings + cross-layer KV reuse (gemma-4-E2B-it features): 4 layers
# [sliding(own kv, HAS attn_v), full(own kv, NO attn_v: use_v_from_k), sliding(shared, reuses layer
# 0's kv), full(shared, reuses layer 1's kv)]; n_kv_shared_layers=2 puts the provider/shared split
# at n_layer_kv_from_start=2, and layer 1 (full, non-shared) deliberately has NO attn_v.weight,
# opposite of the plain fixture above, proving `use_v_from_k` follows per-layer tensor presence, not
# an is_sliding proxy. Shared layers get double-width ffn (`use_double_wide_mlp`, a plain per-layer
# array).
PLE_N_LAYER = 4
PLE_PER_LAYER_DIM = 4
PLE_N_KV_SHARED_LAYERS = 2
PLE_FFN_LEN_BASE = 8
_PLE_IS_SLIDING = [True, False, True, False]
_PLE_HEAD_DIM = [HEAD_DIM_LOCAL, HEAD_DIM_GLOBAL, HEAD_DIM_LOCAL, HEAD_DIM_GLOBAL]
_PLE_N_HEAD_KV = [N_HEAD_KV_LOCAL, N_HEAD_KV_GLOBAL, N_HEAD_KV_LOCAL, N_HEAD_KV_GLOBAL]
_PLE_HAS_OWN_KV = [True, True, False, False]
_PLE_HAS_ATTN_V = [True, False]


def build_tiny_gemma4_ple_kv_shared_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [6] * 256
    vocab_size = len(tokens)
    ffn_len_arr = [PLE_FFN_LEN_BASE * (1 if own_kv else 2) for own_kv in _PLE_HAS_OWN_KV]

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "gemma4")
        .set_u32("gemma4.embedding_length", N_EMBD)
        .set_u32("gemma4.attention.head_count", N_HEAD)
        .set_array("gemma4.attention.head_count_kv", GGUFValueType.UINT32, _PLE_N_HEAD_KV)
        .set_array("gemma4.attention.sliding_window_pattern", GGUFValueType.BOOL, _PLE_IS_SLIDING)
        .set_u32("gemma4.attention.sliding_window", SLIDING_WINDOW)
        .set_u32("gemma4.attention.key_length", HEAD_DIM_GLOBAL)
        .set_u32("gemma4.attention.key_length_swa", HEAD_DIM_LOCAL)
        .set_u32("gemma4.attention.shared_kv_layers", PLE_N_KV_SHARED_LAYERS)
        .set_u32("gemma4.block_count", PLE_N_LAYER)
        .set_array("gemma4.feed_forward_length", GGUFValueType.UINT32, ffn_len_arr)
        .set_u32("gemma4.embedding_length_per_layer_input", PLE_PER_LAYER_DIM)
        .set_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
        .set_f32("gemma4.rope.freq_base", 1000000.0)
        .set_f32("gemma4.rope.freq_base_swa", 10000.0)
        .set_str("tokenizer.ggml.model", "gemma4")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, [0.0] * vocab_size)
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
    add("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add(
        "per_layer_token_embd.weight",
        _random_weight(rng, vocab_size, PLE_N_LAYER * PLE_PER_LAYER_DIM),
    )
    add("per_layer_model_proj.weight", _random_weight(rng, PLE_N_LAYER * PLE_PER_LAYER_DIM, N_EMBD))
    add("per_layer_proj_norm.weight", np.ones(PLE_PER_LAYER_DIM, dtype="<f4"))

    for i in range(PLE_N_LAYER):
        head_dim = _PLE_HEAD_DIM[i]
        n_head_kv = _PLE_N_HEAD_KV[i]
        ffn_len = ffn_len_arr[i]
        prefix = f"blk.{i}."

        add(prefix + "attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "attn_q.weight", _random_weight(rng, N_HEAD * head_dim, N_EMBD))
        add(prefix + "attn_q_norm.weight", np.ones(head_dim, dtype="<f4"))
        if _PLE_HAS_OWN_KV[i]:
            add(prefix + "attn_k.weight", _random_weight(rng, n_head_kv * head_dim, N_EMBD))
            add(prefix + "attn_k_norm.weight", np.ones(head_dim, dtype="<f4"))
            if _PLE_HAS_ATTN_V[i]:
                add(prefix + "attn_v.weight", _random_weight(rng, n_head_kv * head_dim, N_EMBD))
        add(prefix + "attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * head_dim))
        add(prefix + "post_attention_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_gate.weight", _random_weight(rng, ffn_len, N_EMBD))
        add(prefix + "ffn_up.weight", _random_weight(rng, ffn_len, N_EMBD))
        add(prefix + "ffn_down.weight", _random_weight(rng, N_EMBD, ffn_len))
        add(prefix + "post_ffw_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "layer_output_scale.weight", np.ones(1, dtype="<f4"))
        add(prefix + "inp_gate.weight", _random_weight(rng, PLE_PER_LAYER_DIM, N_EMBD))
        add(prefix + "proj.weight", _random_weight(rng, N_EMBD, PLE_PER_LAYER_DIM))
        add(prefix + "post_norm.weight", np.ones(N_EMBD, dtype="<f4"))

    return builder.write(path)


# Mixture-of-Experts (gemma-4-*-A*B-it feature,; gemma4_moe.py): no real MoE GGUF exists,
# so this proves the wiring/op order only. 2 plain uniform (non-sliding) layers, separate from the
# PLE/kv-sharing fixture above so each mechanism's test surface stays isolated.
MOE_N_LAYER = 2
MOE_HEAD_DIM = 4
MOE_N_HEAD_KV = 2
MOE_FFN_LEN = 16
MOE_NUM_EXPERTS = 4
MOE_NUM_EXPERTS_PER_TOK = 2
MOE_EXPERT_FFN_LEN = 12


def build_tiny_gemma4_moe_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [6] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "gemma4")
        .set_u32("gemma4.embedding_length", N_EMBD)
        .set_u32("gemma4.attention.head_count", N_HEAD)
        .set_array(
            "gemma4.attention.head_count_kv", GGUFValueType.UINT32, [MOE_N_HEAD_KV] * MOE_N_LAYER
        )
        .set_array(
            "gemma4.attention.sliding_window_pattern", GGUFValueType.BOOL, [False] * MOE_N_LAYER
        )
        .set_u32("gemma4.attention.sliding_window", SLIDING_WINDOW)
        .set_u32("gemma4.attention.key_length", MOE_HEAD_DIM)
        .set_u32("gemma4.block_count", MOE_N_LAYER)
        .set_u32("gemma4.feed_forward_length", MOE_FFN_LEN)
        .set_u32("gemma4.expert_count", MOE_NUM_EXPERTS)
        .set_u32("gemma4.expert_used_count", MOE_NUM_EXPERTS_PER_TOK)
        .set_u32("gemma4.expert_feed_forward_length", MOE_EXPERT_FFN_LEN)
        .set_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
        .set_f32("gemma4.rope.freq_base", 1000000.0)
        .set_f32("gemma4.rope.freq_base_swa", 10000.0)
        .set_str("tokenizer.ggml.model", "gemma4")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.scores", GGUFValueType.FLOAT32, [0.0] * vocab_size)
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
    add("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))

    for i in range(MOE_N_LAYER):
        prefix = f"blk.{i}."
        add(prefix + "attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "attn_q.weight", _random_weight(rng, N_HEAD * MOE_HEAD_DIM, N_EMBD))
        add(prefix + "attn_q_norm.weight", np.ones(MOE_HEAD_DIM, dtype="<f4"))
        add(prefix + "attn_k.weight", _random_weight(rng, MOE_N_HEAD_KV * MOE_HEAD_DIM, N_EMBD))
        add(prefix + "attn_k_norm.weight", np.ones(MOE_HEAD_DIM, dtype="<f4"))
        add(prefix + "attn_v.weight", _random_weight(rng, MOE_N_HEAD_KV * MOE_HEAD_DIM, N_EMBD))
        add(prefix + "attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * MOE_HEAD_DIM))
        add(prefix + "post_attention_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_gate.weight", _random_weight(rng, MOE_FFN_LEN, N_EMBD))
        add(prefix + "ffn_up.weight", _random_weight(rng, MOE_FFN_LEN, N_EMBD))
        add(prefix + "ffn_down.weight", _random_weight(rng, N_EMBD, MOE_FFN_LEN))
        add(prefix + "post_ffw_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "layer_output_scale.weight", np.ones(1, dtype="<f4"))
        # MoE tensors
        add(prefix + "ffn_gate_inp.weight", _random_weight(rng, MOE_NUM_EXPERTS, N_EMBD))
        add(prefix + "ffn_gate_inp.scale", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_pre_norm_2.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_post_norm_1.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_post_norm_2.weight", np.ones(N_EMBD, dtype="<f4"))
        add(
            prefix + "ffn_gate_exps.weight",
            _random_weight(rng, MOE_NUM_EXPERTS * MOE_EXPERT_FFN_LEN, N_EMBD).reshape(
                MOE_NUM_EXPERTS, MOE_EXPERT_FFN_LEN, N_EMBD
            ),
        )
        add(
            prefix + "ffn_up_exps.weight",
            _random_weight(rng, MOE_NUM_EXPERTS * MOE_EXPERT_FFN_LEN, N_EMBD).reshape(
                MOE_NUM_EXPERTS, MOE_EXPERT_FFN_LEN, N_EMBD
            ),
        )
        add(
            prefix + "ffn_down_exps.weight",
            _random_weight(rng, MOE_NUM_EXPERTS * N_EMBD, MOE_EXPERT_FFN_LEN).reshape(
                MOE_NUM_EXPERTS, N_EMBD, MOE_EXPERT_FFN_LEN
            ),
        )
        add(prefix + "ffn_down_exps.scale", np.ones(MOE_NUM_EXPERTS, dtype="<f4"))

    return builder.write(path)

"""Builds a tiny, complete, valid `gemma4` GGUF file - real architecture

shape (including both local/sliding and global layer types in one model,
different head_dim per type, the `rope_freqs.weight` correction tensor,
and final-logit softcapping), real (if minimal) Gemma4Tokenizer-shaped
metadata, all F32 - so integration tests can run the *entire* real pipeline
(GGUFReader -> GGUFModelLoader -> Gemma4Architecture -> Gemma4Tokenizer ->
ChatEngine -> NDJSON) end to end, fast and memory-safely.

This proves the *wiring* (shapes, per-layer-type dispatch, sliding-window
masking, the v-from-k reuse on the global layer, the rope_freqs code path,
sandwich-norm ordering) doesn't crash and produces correctly-shaped output -
not that it numerically matches the real `google/gemma-4-12b-it` model,
which a much larger real GGUF pull (7GB) makes impractical to fully
validate on this project's 15GB-RAM target hardware even in bf16 (~28GB
estimated) - see ROADMAP.md's M10 gemma4 entry for the accepted gap this
leaves and why.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
N_LAYER = 6  # one full real sliding_window_pattern cycle: 5 local + 1 global
FFN_LEN = 16
SLIDING_WINDOW = 4
HEAD_DIM_LOCAL = 4
HEAD_DIM_GLOBAL = 6  # deliberately different from local, to prove per-layer-type dispatch
N_HEAD_KV_LOCAL = 2  # plain MHA on local layers - no GQA repeat needed
N_HEAD_KV_GLOBAL = 1  # MQA on the global layer - exercises the repeat_interleave path
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
        # GGUF's ne[] is fastest-dim-first (PyTorch's shape reversed) - see
        # the row-permute note in app/architectures/layers.py.
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add("token_embd.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    # Real llama.cpp freq_factors convention - a no-op correction (all 1.0) is enough to
    # exercise the code path without needing to fake a numerically meaningful value.
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

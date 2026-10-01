"""Builds a tiny, complete, valid `qwen35` GGUF: 4 layers (3 Gated DeltaNet + 1 full attention at
`full_attention_interval=4`), fewer key heads than value heads (exercises the head expansion),
`head_dim` != `N_EMBD // N_HEAD`, and `rope.dimension_count` < `head_dim` (partial RoPE). Tensor
names/shapes mirror a real `Qwen3.5-9B` GGUF header (range-fetched 2026-10-01).
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 8  # attention head dim; != N_EMBD // N_HEAD (4)
ROPE_DIM = 4  # < HEAD_DIM: partial RoPE
N_LAYER = 4
FFN_LEN = 16
CONV_KERNEL = 4
SSM_HEAD_DIM = 4
N_K_HEADS = 2
N_V_HEADS = 4
CONV_DIM = (2 * N_K_HEADS + N_V_HEADS) * SSM_HEAD_DIM
VALUE_DIM = N_V_HEADS * SSM_HEAD_DIM

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def _byte_tokens() -> list[str]:
    from app.runtime.tokenizer import _byte_to_unicode

    byte_encoder = _byte_to_unicode()
    return [byte_encoder[b] for b in range(256)]


def build_tiny_qwen35_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
    rng = np.random.RandomState(seed)
    tokens = ["<unk>", "<s>", "</s>"] + _byte_tokens()
    token_types = [3] * 3 + [1] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "qwen35")
        .set_u32("qwen35.embedding_length", N_EMBD)
        .set_u32("qwen35.attention.head_count", N_HEAD)
        .set_u32("qwen35.attention.head_count_kv", N_HEAD_KV)
        .set_u32("qwen35.attention.key_length", HEAD_DIM)
        .set_u32("qwen35.block_count", N_LAYER)
        .set_u32("qwen35.feed_forward_length", FFN_LEN)
        .set_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        .set_f32("qwen35.rope.freq_base", 10000.0)
        .set_u32("qwen35.rope.dimension_count", ROPE_DIM)
        .set_u32("qwen35.full_attention_interval", 4)
        .set_u32("qwen35.ssm.conv_kernel", CONV_KERNEL)
        .set_u32("qwen35.ssm.state_size", SSM_HEAD_DIM)
        .set_u32("qwen35.ssm.group_count", N_K_HEADS)
        .set_u32("qwen35.ssm.time_step_rank", N_V_HEADS)
        .set_u32("qwen35.ssm.inner_size", VALUE_DIM)
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

    def weight(out_features: int, in_features: int) -> np.ndarray:
        return (rng.randn(out_features, in_features) * 0.1).astype("<f4")

    def norm(dim: int) -> np.ndarray:
        return (np.ones(dim) + rng.randn(dim) * 0.05).astype("<f4")

    add("token_embd.weight", weight(vocab_size, N_EMBD))
    if not tied_embeddings:
        add("output.weight", weight(vocab_size, N_EMBD))
    add("output_norm.weight", norm(N_EMBD))
    for i in range(N_LAYER):
        p = f"blk.{i}."
        add(p + "attn_norm.weight", norm(N_EMBD))
        add(p + "post_attention_norm.weight", norm(N_EMBD))
        add(p + "ffn_gate.weight", weight(FFN_LEN, N_EMBD))
        add(p + "ffn_up.weight", weight(FFN_LEN, N_EMBD))
        add(p + "ffn_down.weight", weight(N_EMBD, FFN_LEN))
        if (i + 1) % 4 == 0:
            add(p + "attn_q.weight", weight(N_HEAD * HEAD_DIM * 2, N_EMBD))
            add(p + "attn_k.weight", weight(N_HEAD_KV * HEAD_DIM, N_EMBD))
            add(p + "attn_v.weight", weight(N_HEAD_KV * HEAD_DIM, N_EMBD))
            add(p + "attn_output.weight", weight(N_EMBD, N_HEAD * HEAD_DIM))
            add(p + "attn_q_norm.weight", norm(HEAD_DIM))
            add(p + "attn_k_norm.weight", norm(HEAD_DIM))
        else:
            add(p + "attn_qkv.weight", weight(CONV_DIM, N_EMBD))
            add(p + "attn_gate.weight", weight(VALUE_DIM, N_EMBD))
            add(p + "ssm_beta.weight", weight(N_V_HEADS, N_EMBD))
            add(p + "ssm_alpha.weight", weight(N_V_HEADS, N_EMBD))
            add(p + "ssm_out.weight", weight(N_EMBD, VALUE_DIM))
            add(p + "ssm_conv1d.weight", weight(CONV_DIM, CONV_KERNEL))
            add(p + "ssm_dt.bias", (rng.randn(N_V_HEADS) * 0.1).astype("<f4"))
            add(p + "ssm_a", (-rng.rand(N_V_HEADS) * 2 - 0.01).astype("<f4"))
            add(p + "ssm_norm.weight", norm(SSM_HEAD_DIM))
    return builder.write(path)

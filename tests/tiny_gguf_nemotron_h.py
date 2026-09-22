"""Builds a tiny, complete, valid `nemotron_h` GGUF file - mirrors `tiny_gguf_granitemoe.py`'s
structure, plus the real per-layer-array metadata (`nemotron_h.attention.head_count_kv`/
`nemotron_h.feed_forward_length`, one value per layer, 0 for layers that don't have that
component) that encodes which of the three real layer types (Mamba-2/attention/MLP) each layer
actually is - see `NemotronHArchitecture`'s own docstring for the real derivation logic this
fixture exercises. 6 tiny layers cover all three types at least twice (Mamba/MLP) or once
(attention), matching the real target checkpoint's dominant-Mamba shape.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
HEAD_DIM = 4
N_HEAD_KV = 2
FFN_LEN = 16
MAMBA_NUM_HEADS = 2
MAMBA_HEAD_DIM = 4
MAMBA_D_INNER = MAMBA_NUM_HEADS * MAMBA_HEAD_DIM  # 8
D_STATE = 4
N_GROUP = 2
CONV_KERNEL = 4

LAYER_TYPES = ["mamba", "attention", "mlp", "mamba", "mlp", "mamba"]
N_LAYER = len(LAYER_TYPES)

_CONTROL_TOKENS = ["<unk>", "<s>", "</s>"]
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def _byte_tokens() -> list[str]:
    from app.runtime.tokenizer import _byte_to_unicode

    byte_encoder = _byte_to_unicode()
    return [byte_encoder[b] for b in range(256)]


def _random_weight(rng: np.random.RandomState, *shape: int) -> np.ndarray:
    return (rng.randn(*shape) * 0.02).astype("<f4")


def build_tiny_nemotron_h_gguf(path: Path, seed: int = 0, tied_embeddings: bool = False) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [1] * 256
    vocab_size = len(tokens)
    conv_dim = MAMBA_D_INNER + 2 * N_GROUP * D_STATE
    in_proj_out = 2 * MAMBA_D_INNER + 2 * N_GROUP * D_STATE + MAMBA_NUM_HEADS

    head_count_kv = [N_HEAD_KV if t == "attention" else 0 for t in LAYER_TYPES]
    ffn_len_arr = [FFN_LEN if t == "mlp" else 0 for t in LAYER_TYPES]

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "nemotron_h")
        .set_u32("nemotron_h.embedding_length", N_EMBD)
        .set_u32("nemotron_h.attention.head_count", N_HEAD)
        .set_u32("nemotron_h.block_count", N_LAYER)
        .set_f32("nemotron_h.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("nemotron_h.vocab_size", vocab_size)
        .set_array("nemotron_h.attention.head_count_kv", GGUFValueType.UINT32, head_count_kv)
        .set_array("nemotron_h.feed_forward_length", GGUFValueType.UINT32, ffn_len_arr)
        .set_u32("nemotron_h.ssm.conv_kernel", CONV_KERNEL)
        .set_u32("nemotron_h.ssm.inner_size", MAMBA_D_INNER)
        .set_u32("nemotron_h.ssm.state_size", D_STATE)
        .set_u32("nemotron_h.ssm.time_step_rank", MAMBA_NUM_HEADS)
        .set_u32("nemotron_h.ssm.group_count", N_GROUP)
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

    for i, layer_type in enumerate(LAYER_TYPES):
        add(f"blk.{i}.attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        if layer_type == "mamba":
            add(f"blk.{i}.ssm_in.weight", _random_weight(rng, in_proj_out, N_EMBD))
            add(f"blk.{i}.ssm_conv1d.weight", _random_weight(rng, conv_dim, CONV_KERNEL))
            add(f"blk.{i}.ssm_conv1d.bias", np.zeros(conv_dim, dtype="<f4"))
            add(f"blk.{i}.ssm_dt.bias", np.zeros(MAMBA_NUM_HEADS, dtype="<f4"))
            add(f"blk.{i}.ssm_a", -np.abs(_random_weight(rng, MAMBA_NUM_HEADS)) - 0.01)
            add(f"blk.{i}.ssm_d", np.ones(MAMBA_NUM_HEADS, dtype="<f4"))
            add(
                f"blk.{i}.ssm_norm.weight",
                np.ones((N_GROUP, MAMBA_D_INNER // N_GROUP), dtype="<f4"),
            )
            add(f"blk.{i}.ssm_out.weight", _random_weight(rng, N_EMBD, MAMBA_D_INNER))
        elif layer_type == "attention":
            add(f"blk.{i}.attn_q.weight", _random_weight(rng, N_HEAD * HEAD_DIM, N_EMBD))
            add(f"blk.{i}.attn_k.weight", _random_weight(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
            add(f"blk.{i}.attn_v.weight", _random_weight(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
            add(f"blk.{i}.attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * HEAD_DIM))
        else:
            add(f"blk.{i}.ffn_up.weight", _random_weight(rng, FFN_LEN, N_EMBD))
            add(f"blk.{i}.ffn_down.weight", _random_weight(rng, N_EMBD, FFN_LEN))

    return builder.write(path)

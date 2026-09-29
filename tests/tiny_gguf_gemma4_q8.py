"""A tiny `gemma4` GGUF whose two embedding tables (`token_embd.weight`, gemma4's own tied
lm_head, and `per_layer_token_embd.weight`, its real Per-Layer Embeddings table - see
gemma4_ple.py) are Q8_0-quantized, everything else F32 - big enough (n_embd 32, one Q8_0 block)
for the packed `QuantizedEmbedding` path (app/architectures/quantized_embedding.py) to be checked
end to end against the float path on the exact same quantized values, same precedent
tests/tiny_gguf_llama_q8.py already established. One plain, non-sliding, non-KV-shared layer -
tests/tiny_gguf_gemma4.py's own PLE fixture already covers the sliding/KV-sharing combination
with different (Q8_0-incompatible, n_embd=8) dimensions, so this stays deliberately separate
rather than risking those already-passing tests over unrelated dimension changes.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder
from tests.tiny_gguf_llama_q8 import _q8_0_bytes

N_EMBD = 32
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM = 16  # N_HEAD * HEAD_DIM == N_EMBD, same real relationship every actual checkpoint has
N_LAYER = 1
FFN_LEN = 32
PER_LAYER_DIM = 32  # N_LAYER * PER_LAYER_DIM must also be a Q8_0-block-aligned (32) width
FINAL_LOGIT_SOFTCAPPING = 30.0

_CONTROL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 1


def _byte_tokens() -> list[str]:
    return [f"<0x{b:02X}>" for b in range(256)]


def _random_f32(rng: np.random.RandomState, *shape: int) -> np.ndarray:
    return (rng.randn(*shape) * 0.02).astype("<f4")


def build_tiny_gemma4_ple_q8_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [6] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "gemma4")
        .set_u32("gemma4.embedding_length", N_EMBD)
        .set_u32("gemma4.attention.head_count", N_HEAD)
        .set_u32("gemma4.attention.head_count_kv", N_HEAD_KV)
        .set_u32("gemma4.attention.key_length", HEAD_DIM)
        .set_u32("gemma4.block_count", N_LAYER)
        .set_u32("gemma4.feed_forward_length", FFN_LEN)
        .set_u32("gemma4.embedding_length_per_layer_input", PER_LAYER_DIM)
        .set_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6)
        .set_f32("gemma4.final_logit_softcapping", FINAL_LOGIT_SOFTCAPPING)
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

    def add_f32(name: str, array: np.ndarray) -> None:
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    def add_q8(name: str, out_features: int, in_features: int) -> None:
        weight = rng.randn(out_features, in_features) * 0.1
        builder.add_tensor(
            name, [in_features, out_features], GGMLQuantizationType.Q8_0, _q8_0_bytes(weight)
        )

    add_q8("token_embd.weight", vocab_size, N_EMBD)
    add_f32("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_q8("per_layer_token_embd.weight", vocab_size, N_LAYER * PER_LAYER_DIM)
    add_f32("per_layer_model_proj.weight", _random_f32(rng, N_LAYER * PER_LAYER_DIM, N_EMBD))
    add_f32("per_layer_proj_norm.weight", np.ones(PER_LAYER_DIM, dtype="<f4"))

    prefix = "blk.0."
    add_f32(prefix + "attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32(prefix + "attn_q.weight", _random_f32(rng, N_HEAD * HEAD_DIM, N_EMBD))
    add_f32(prefix + "attn_q_norm.weight", np.ones(HEAD_DIM, dtype="<f4"))
    add_f32(prefix + "attn_k.weight", _random_f32(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
    add_f32(prefix + "attn_k_norm.weight", np.ones(HEAD_DIM, dtype="<f4"))
    add_f32(prefix + "attn_v.weight", _random_f32(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
    add_f32(prefix + "attn_output.weight", _random_f32(rng, N_EMBD, N_HEAD * HEAD_DIM))
    add_f32(prefix + "post_attention_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32(prefix + "ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32(prefix + "ffn_gate.weight", _random_f32(rng, FFN_LEN, N_EMBD))
    add_f32(prefix + "ffn_up.weight", _random_f32(rng, FFN_LEN, N_EMBD))
    add_f32(prefix + "ffn_down.weight", _random_f32(rng, N_EMBD, FFN_LEN))
    add_f32(prefix + "post_ffw_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add_f32(prefix + "layer_output_scale.weight", np.ones(1, dtype="<f4"))
    add_f32(prefix + "inp_gate.weight", _random_f32(rng, PER_LAYER_DIM, N_EMBD))
    add_f32(prefix + "proj.weight", _random_f32(rng, N_EMBD, PER_LAYER_DIM))
    add_f32(prefix + "post_norm.weight", np.ones(N_EMBD, dtype="<f4"))

    return builder.write(path)

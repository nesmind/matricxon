"""Builds a tiny, complete, valid `phi2` GGUF file - real architecture, real gpt2-BPE tokenizer
metadata, all F32 - so integration tests can run the *entire* real pipeline (GGUFReader ->
GGUFModelLoader -> ArchitectureRegistry -> Phi2Architecture -> GGUFTokenizer -> ChatEngine ->
NDJSON) end to end, fast and memory-safely. The generated text is gibberish (tiny near-random
weights) - this exercises the wiring (including the real fused-qkv split and partial-rotary
shapes, both real Phi-2 novelties - see app/architectures/phi2.py's own docstring), not model
quality; a real moondream2 pull is what validates the actual math (see
tests/integration/test_real_fixture_moondream2.py).

ROPE_DIM < HEAD_DIM on purpose (2 of 4, the same 1:2 ratio moondream2's own real GGUF uses - 32 of
64) so this fixture actually exercises `apply_rotary_pos_emb_partial`'s split/concat path, not just
a degenerate full-head case.
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
HEAD_DIM = 4
ROPE_DIM = 2
N_LAYER = 1
FFN_LEN = 16

_CONTROL_TOKENS = ["<|endoftext|>"]
EOS_TOKEN_ID = 0


def _byte_tokens() -> list[str]:
    from app.runtime.tokenizer import _byte_to_unicode

    byte_encoder = _byte_to_unicode()
    return [byte_encoder[b] for b in range(256)]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def _random_bias(rng: np.random.RandomState, out_features: int) -> np.ndarray:
    return (rng.randn(out_features) * 0.02).astype("<f4")


def build_tiny_phi2_gguf(path: Path, seed: int = 0, n_layer: int = N_LAYER) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [1] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "phi2")
        .set_u32("phi2.embedding_length", N_EMBD)
        .set_u32("phi2.attention.head_count", N_HEAD)
        .set_u32("phi2.attention.head_count_kv", N_HEAD)
        .set_u32("phi2.block_count", n_layer)
        .set_u32("phi2.feed_forward_length", FFN_LEN)
        .set_f32("phi2.attention.layer_norm_epsilon", 1e-5)
        .set_u32("phi2.rope.dimension_count", ROPE_DIM)
        .set_str("tokenizer.ggml.model", "gpt2")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.merges", GGUFValueType.STRING, [])
        .set_array("tokenizer.ggml.token_type", GGUFValueType.INT32, token_types)
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
    add("output_norm.bias", np.zeros(N_EMBD, dtype="<f4"))
    add("output.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("output.bias", _random_bias(rng, vocab_size))
    for i in range(n_layer):
        prefix = f"blk.{i}."
        add(prefix + "attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "attn_norm.bias", np.zeros(N_EMBD, dtype="<f4"))
        add(prefix + "attn_qkv.weight", _random_weight(rng, 3 * N_EMBD, N_EMBD))
        add(prefix + "attn_qkv.bias", _random_bias(rng, 3 * N_EMBD))
        add(prefix + "attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * HEAD_DIM))
        add(prefix + "attn_output.bias", _random_bias(rng, N_EMBD))
        add(prefix + "ffn_up.weight", _random_weight(rng, FFN_LEN, N_EMBD))
        add(prefix + "ffn_up.bias", _random_bias(rng, FFN_LEN))
        add(prefix + "ffn_down.weight", _random_weight(rng, N_EMBD, FFN_LEN))
        add(prefix + "ffn_down.bias", _random_bias(rng, N_EMBD))

    return builder.write(path)

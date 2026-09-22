"""Builds a tiny, complete, valid `mistral3` GGUF file - real architecture,
real tokenizer metadata, all F32 (no quantization needed at this size) -
so integration tests can run the *entire* real pipeline (GGUFReader ->
GGUFModelLoader -> ArchitectureRegistry -> Mistral3TextArchitecture ->
GGUFTokenizer -> ChatEngine -> NDJSON) end to end, fast and memory-safely.
The generated text is gibberish (tiny near-random weights) - this is for
exercising the wiring, not model quality; ROADMAP.md's M3/M4 already
validated the real forward-pass math against the real weights.
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

# Control tokens (matching what Mistral3PromptBuilder/GGUFTokenizer expect,
# including the M10 tool-calling ones - confirmed as real control tokens via
# mistralai/Ministral-3-3B-Instruct-2512's real tokenizer_config.json
# `extra_special_tokens`, not assumed) followed by one token per byte value
# (0-255), covering every possible UTF-8 byte sequence without needing any
# real BPE merges.
_CONTROL_TOKENS = [
    "<unk>",
    "<s>",
    "</s>",
    "[INST]",
    "[/INST]",
    "[SYSTEM_PROMPT]",
    "[/SYSTEM_PROMPT]",
    "[IMG]",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_CALLS]",
    "[ARGS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
]
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2


def _byte_tokens() -> list[str]:
    from app.runtime.tokenizer import _byte_to_unicode

    byte_encoder = _byte_to_unicode()
    return [byte_encoder[b] for b in range(256)]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def build_tiny_mistral3_gguf(path: Path, seed: int = 0, n_layer: int = N_LAYER) -> Path:
    """`n_layer` defaults to the module's own single-layer N_LAYER (every existing caller gets
    identical output to before this param existed) - only tests that specifically need a multi-
    layer model (e.g. mixed per-layer dtype planning, see tests/unit/test_model_manager_memory.py)
    pass a larger value."""
    rng = np.random.RandomState(seed)
    tokens = _CONTROL_TOKENS + _byte_tokens()
    token_types = [3] * len(_CONTROL_TOKENS) + [1] * 256
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "mistral3")
        .set_u32("mistral3.embedding_length", N_EMBD)
        .set_u32("mistral3.attention.head_count", N_HEAD)
        .set_u32("mistral3.attention.head_count_kv", N_HEAD_KV)
        .set_u32("mistral3.attention.key_length", HEAD_DIM)
        .set_u32("mistral3.block_count", n_layer)
        .set_u32("mistral3.feed_forward_length", FFN_LEN)
        .set_f32("mistral3.attention.layer_norm_rms_epsilon", 1e-5)
        .set_u32("mistral3.vocab_size", vocab_size)
        .set_f32("mistral3.rope.freq_base", 10000.0)
        .set_f32("mistral3.rope.scaling.factor", 1.0)
        .set_f32("mistral3.rope.scaling.beta_fast", 32.0)
        .set_f32("mistral3.rope.scaling.beta_slow", 1.0)
        .set_u32("mistral3.rope.scaling.original_context_length", 16)
        .set_f32("mistral3.rope.scaling.mscale", 1.0)
        .set_f32("mistral3.rope.scaling.mscale_all_dim", 1.0)
        .set_str("tokenizer.ggml.model", "gpt2")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_array("tokenizer.ggml.merges", GGUFValueType.STRING, [])
        .set_array("tokenizer.ggml.token_type", GGUFValueType.INT32, token_types)
        .set_u32("tokenizer.ggml.bos_token_id", BOS_TOKEN_ID)
        .set_u32("tokenizer.ggml.eos_token_id", EOS_TOKEN_ID)
    )

    def add(name: str, array: np.ndarray) -> None:
        # GGUF's ne[] is fastest-dim-first (PyTorch's shape reversed) - see
        # the row-permute note in app/architectures/mistral3.py.
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add("token_embd.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("output_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    for i in range(n_layer):
        prefix = f"blk.{i}."
        add(prefix + "attn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ffn_norm.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "attn_q.weight", _random_weight(rng, N_HEAD * HEAD_DIM, N_EMBD))
        add(prefix + "attn_k.weight", _random_weight(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
        add(prefix + "attn_v.weight", _random_weight(rng, N_HEAD_KV * HEAD_DIM, N_EMBD))
        add(prefix + "attn_output.weight", _random_weight(rng, N_EMBD, N_HEAD * HEAD_DIM))
        add(prefix + "ffn_gate.weight", _random_weight(rng, FFN_LEN, N_EMBD))
        add(prefix + "ffn_up.weight", _random_weight(rng, FFN_LEN, N_EMBD))
        add(prefix + "ffn_down.weight", _random_weight(rng, N_EMBD, FFN_LEN))

    return builder.write(path)

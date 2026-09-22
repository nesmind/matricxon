"""Builds a tiny, complete, valid `bert` GGUF file - the WordPiece/encoder
equivalent of tiny_gguf.py's `build_tiny_mistral3_gguf`. Real architecture,
real (if minimal) WordPiece-shaped tokenizer metadata, all F32, so
integration tests can run the entire real embeddings pipeline
(GGUFReader -> GGUFModelLoader -> BertArchitecture -> WordPieceTokenizer ->
EmbeddingEngine) end to end, fast and memory-safely. ROADMAP.md's M8 oracle
scripts already validated the real forward-pass math against real weights -
this is for exercising the wiring, not model quality.
"""

import string
from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
N_LAYER = 1
FFN_LEN = 16
MAX_POSITION = 16

_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
UNK_TOKEN_ID = 1
CLS_TOKEN_ID = 2
SEP_TOKEN_ID = 3


def _wordpiece_vocab() -> list[str]:
    # A word-initial ("▁"-prefixed) and a bare (continuation) entry for
    # every lowercase ASCII letter - enough to encode any short lowercase
    # test word one character at a time without ever hitting [UNK].
    letters = list(string.ascii_lowercase)
    return _SPECIAL_TOKENS + [f"▁{c}" for c in letters] + letters


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def build_tiny_bert_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    tokens = _wordpiece_vocab()
    vocab_size = len(tokens)

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "bert")
        .set_u32("bert.embedding_length", N_EMBD)
        .set_u32("bert.attention.head_count", N_HEAD)
        .set_u32("bert.block_count", N_LAYER)
        .set_u32("bert.feed_forward_length", FFN_LEN)
        .set_f32("bert.attention.layer_norm_epsilon", 1e-12)
        .set_u32("bert.context_length", MAX_POSITION)
        .set_str("tokenizer.ggml.model", "bert")
        .set_array("tokenizer.ggml.tokens", GGUFValueType.STRING, tokens)
        .set_u32("tokenizer.ggml.unknown_token_id", UNK_TOKEN_ID)
        .set_u32("tokenizer.ggml.cls_token_id", CLS_TOKEN_ID)
        .set_u32("tokenizer.ggml.seperator_token_id", SEP_TOKEN_ID)
    )

    def add(name: str, array: np.ndarray) -> None:
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add("token_embd.weight", _random_weight(rng, vocab_size, N_EMBD))
    add("position_embd.weight", _random_weight(rng, MAX_POSITION, N_EMBD))
    add("token_types.weight", _random_weight(rng, 2, N_EMBD))
    add("token_embd_norm.weight", np.ones(N_EMBD, dtype="<f4"))
    add("token_embd_norm.bias", np.zeros(N_EMBD, dtype="<f4"))

    for name, out_dim, in_dim in [
        ("blk.0.attn_q.weight", N_EMBD, N_EMBD),
        ("blk.0.attn_k.weight", N_EMBD, N_EMBD),
        ("blk.0.attn_v.weight", N_EMBD, N_EMBD),
        ("blk.0.attn_output.weight", N_EMBD, N_EMBD),
        ("blk.0.ffn_up.weight", FFN_LEN, N_EMBD),
        ("blk.0.ffn_down.weight", N_EMBD, FFN_LEN),
    ]:
        add(name, _random_weight(rng, out_dim, in_dim))
        add(name.replace(".weight", ".bias"), np.zeros(out_dim, dtype="<f4"))

    for name in ["blk.0.attn_output_norm", "blk.0.layer_output_norm"]:
        add(f"{name}.weight", np.ones(N_EMBD, dtype="<f4"))
        add(f"{name}.bias", np.zeros(N_EMBD, dtype="<f4"))

    return builder.write(path)

"""Builds a tiny, complete, valid `clip` mmproj GGUF using the real

Idefics3/SmolVLM projector shape (SigLIP tower, no CLS token, no pre_ln, a
real post_ln, a pixel-shuffle + single bias-free linear projector) - see
tests/tiny_gguf_clip.py's own docstring for why a synthetic fixture is used
instead of the real 175MB SmolVLM download for fast unit tests. The real
numeric/statistical validation against the real
`ggml-org/SmolVLM-256M-Instruct-GGUF` mmproj pull is done manually (see
ROADMAP.md's vision-fusion entry).
"""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from scripts.make_tiny_gguf import GGUFBuilder

N_EMBD = 8
N_HEAD = 2
N_LAYER = 2
FFN_LEN = 16
PATCH_SIZE = 4
IMAGE_SIZE = 16  # -> (16/4)^2 = 16 raw patches (4x4 grid)
SCALE_FACTOR = 2  # -> 16 / (2*2) = 4 output patches (2x2 grid)
PROJECTION_DIM = 6
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]


def _random_weight(rng: np.random.RandomState, out_features: int, in_features: int) -> np.ndarray:
    return (rng.randn(out_features, in_features) * 0.02).astype("<f4")


def build_tiny_idefics3_gguf(path: Path, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    raw_num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

    builder = (
        GGUFBuilder()
        .set_str("general.architecture", "clip")
        .set_bool("clip.has_vision_encoder", True)
        .set_bool("clip.has_text_encoder", False)
        .set_str("clip.projector_type", "idefics3")
        .set_bool("clip.use_gelu", True)
        .set_u32("clip.vision.embedding_length", N_EMBD)
        .set_u32("clip.vision.attention.head_count", N_HEAD)
        .set_u32("clip.vision.block_count", N_LAYER)
        .set_u32("clip.vision.feed_forward_length", FFN_LEN)
        .set_f32("clip.vision.attention.layer_norm_epsilon", 1e-6)
        .set_u32("clip.vision.patch_size", PATCH_SIZE)
        .set_u32("clip.vision.image_size", IMAGE_SIZE)
        .set_u32("clip.vision.projector.scale_factor", SCALE_FACTOR)
        .set_array("clip.vision.image_mean", GGUFValueType.FLOAT32, IMAGE_MEAN)
        .set_array("clip.vision.image_std", GGUFValueType.FLOAT32, IMAGE_STD)
    )

    def add(name: str, array: np.ndarray) -> None:
        # GGUF's ne[] is fastest-dim-first (PyTorch's shape reversed) - see
        # the row-permute note in app/architectures/layers.py.
        builder.add_tensor(
            name, list(reversed(array.shape)), GGMLQuantizationType.F32, array.tobytes()
        )

    add(
        "v.patch_embd.weight",
        _random_weight(rng, N_EMBD, 3 * PATCH_SIZE * PATCH_SIZE).reshape(
            N_EMBD, 3, PATCH_SIZE, PATCH_SIZE
        ),
    )
    add("v.patch_embd.bias", np.zeros(N_EMBD, dtype="<f4"))
    add(
        "v.position_embd.weight",
        _random_weight(rng, raw_num_patches, N_EMBD).reshape(1, raw_num_patches, N_EMBD),
    )

    for i in range(N_LAYER):
        prefix = f"v.blk.{i}."
        add(prefix + "ln1.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ln1.bias", np.zeros(N_EMBD, dtype="<f4"))
        add(prefix + "attn_q.weight", _random_weight(rng, N_EMBD, N_EMBD))
        add(prefix + "attn_q.bias", np.zeros(N_EMBD, dtype="<f4"))
        add(prefix + "attn_k.weight", _random_weight(rng, N_EMBD, N_EMBD))
        add(prefix + "attn_k.bias", np.zeros(N_EMBD, dtype="<f4"))
        add(prefix + "attn_v.weight", _random_weight(rng, N_EMBD, N_EMBD))
        add(prefix + "attn_v.bias", np.zeros(N_EMBD, dtype="<f4"))
        add(prefix + "attn_out.weight", _random_weight(rng, N_EMBD, N_EMBD))
        add(prefix + "attn_out.bias", np.zeros(N_EMBD, dtype="<f4"))
        add(prefix + "ln2.weight", np.ones(N_EMBD, dtype="<f4"))
        add(prefix + "ln2.bias", np.zeros(N_EMBD, dtype="<f4"))
        # Same real, confirmed naming swap as tiny_gguf_clip.py's fixture.
        add(prefix + "ffn_down.weight", _random_weight(rng, FFN_LEN, N_EMBD))
        add(prefix + "ffn_down.bias", np.zeros(FFN_LEN, dtype="<f4"))
        add(prefix + "ffn_up.weight", _random_weight(rng, N_EMBD, FFN_LEN))
        add(prefix + "ffn_up.bias", np.zeros(N_EMBD, dtype="<f4"))

    add("v.post_ln.weight", np.ones(N_EMBD, dtype="<f4"))
    add("v.post_ln.bias", np.zeros(N_EMBD, dtype="<f4"))
    # Real, confirmed on the real SmolVLM mmproj: a single bias-free linear, no
    # "mm.model.fc.bias" tensor at all.
    mm_fc_in = N_EMBD * SCALE_FACTOR * SCALE_FACTOR
    add("mm.model.fc.weight", _random_weight(rng, PROJECTION_DIM, mm_fc_in))

    return builder.write(path)

"""Short checks for the Qwen3.5 vision pieces: interleaved M-RoPE, prefill image slicing, image
size snapping, and the ViT's output shape on a tiny random tower."""

import torch

from app.architectures.rope import InterleavedMRopeEmbedding, RotaryEmbedding
from app.gguf.metadata import GGUFMetadata
from app.runtime.chat_engine import ChatEngine
from app.vision.qwen3vl_encoder import Qwen3VLVisionEncoder
from app.vision.qwen3vl_preprocessing import Qwen3VLImagePreprocessor


def test_mrope_with_equal_axes_matches_plain_rope() -> None:
    plain, mrope = RotaryEmbedding(64, 1e7), InterleavedMRopeEmbedding(64, 1e7, [11, 11, 10, 0])
    pos = torch.arange(9)
    for a, b in zip(plain(pos), mrope(pos.expand(3, -1)), strict=True):
        assert torch.allclose(a, b)
    for a, b in zip(plain(pos), mrope(pos), strict=True):
        assert torch.allclose(a, b)


def test_mrope_axis_layout_is_interleaved() -> None:
    axes = InterleavedMRopeEmbedding(64, 1e7, [11, 11, 10, 0]).axis_of_freq.tolist()
    assert axes[:6] == [0, 1, 2, 0, 1, 2] and axes.count(1) == 11 and axes.count(2) == 10
    assert axes[30:] == [0, 1]


def test_mrope_uses_the_axis_a_frequency_belongs_to() -> None:
    rope = InterleavedMRopeEmbedding(8, 10000.0, [1, 1, 1, 0])  # 4 freqs: axes 0,1,2,0
    pos = torch.tensor([[5], [7], [9]])
    cos, _ = rope(pos)
    angles = torch.tensor([5.0, 7.0, 9.0, 5.0]) * rope.inv_freq
    assert torch.allclose(cos[0, :4], angles.cos())


def test_chunked_prefill_hands_each_piece_only_its_own_image_rows() -> None:
    images = [(4, torch.arange(10.0).view(10, 1))]
    first = ChatEngine._images_in(images, 0, 7)
    second = ChatEngine._images_in(images, 7, 20)
    assert first[0][0] == 4 and first[0][1].flatten().tolist() == [0.0, 1.0, 2.0]
    assert second[0][0] == 0 and second[0][1].flatten().tolist() == list(range(3, 10))
    assert ChatEngine._images_in(images, 14, 20) is None


def test_image_size_snaps_to_multiples_and_respects_pixel_bounds() -> None:
    pre = Qwen3VLImagePreprocessor([0.5] * 3, [0.5] * 3, max_pixels=1024 * 1024)
    assert pre.target_size(500, 700) == (512, 704)
    h, w = pre.target_size(4000, 3000)
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 1024 * 1024
    h, w = pre.target_size(40, 40)
    assert h % 32 == 0 and h * w >= 65536


def test_tiny_vision_tower_merges_2x2_into_text_width() -> None:
    meta = GGUFMetadata(
        {
            "general.architecture": "clip",
            "clip.vision.embedding_length": 16,
            "clip.vision.attention.head_count": 2,
            "clip.vision.block_count": 2,
            "clip.vision.feed_forward_length": 32,
            "clip.vision.attention.layer_norm_epsilon": 1e-6,
            "clip.vision.patch_size": 4,
            "clip.vision.spatial_merge_size": 2,
            "clip.vision.image_size": 32,
            "clip.vision.image_mean": [0.5] * 3,
            "clip.vision.image_std": [0.5] * 3,
        }
    )
    torch.manual_seed(0)
    tower = Qwen3VLVisionEncoder(meta, out_dim=24, mlp_hidden=48).eval()
    out = tower(torch.randn(1, 3, 24, 40))  # 6 x 10 patches -> 3 x 5 merged tokens
    assert out.shape == (15, 24) and torch.isfinite(out).all()
    assert tower.merged_grid(24, 40) == (3, 5)

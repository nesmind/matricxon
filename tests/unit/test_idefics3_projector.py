"""Direct unit tests for app.vision.idefics3_projector - isolated from

ClipVisionEncoder's self-attention (which would otherwise entangle every
patch's output with every other patch's, making a targeted "did this
specific merge group move" assertion meaningless at the full-model level).
"""

import torch

from app.vision.idefics3_projector import compute_num_patches, pixel_shuffle


class TestComputeNumPatches:
    def test_divides_by_scale_factor_squared(self) -> None:
        assert compute_num_patches(raw_num_patches=16, scale_factor=2) == 4

    def test_raises_when_not_evenly_divisible(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="not divisible"):
            compute_num_patches(raw_num_patches=15, scale_factor=2)


class TestPixelShuffle:
    def test_merges_a_2x2_spatial_block_in_row_major_order(self) -> None:
        """Hand-verified against HF transformers' own

        `Idefics3Connector.pixel_shuffle` algorithm for a 4x4 patch grid,
        embed_dim=2, scale_factor=2: each output token must be exactly the
        concatenation of its 2x2 spatial block's 4 patch embeddings, in
        row-major (top-left, top-right, bottom-left, bottom-right) order -
        not some other transposed/flattened arrangement that would still
        happen to produce the right output *shape* but scramble which
        patch's features land in which output token.
        """
        height = width = 4
        embed_dim = 2
        scale_factor = 2

        x = torch.zeros(1, height * width, embed_dim)
        for r in range(height):
            for c in range(width):
                seq_idx = r * width + c
                x[0, seq_idx, 0] = r * 10 + c
                x[0, seq_idx, 1] = 100 + r * 10 + c

        out = pixel_shuffle(x, scale_factor)
        assert out.shape == (1, 4, 8)

        for h_group in range(2):
            for w_group in range(2):
                out_idx = h_group * 2 + w_group
                expected = []
                for dr in range(2):
                    for dc in range(2):
                        r, c = 2 * h_group + dr, 2 * w_group + dc
                        expected.extend([r * 10 + c, 100 + r * 10 + c])
                assert out[0, out_idx].tolist() == expected

    def test_output_shape_matches_batch_and_merged_dims(self) -> None:
        x = torch.randn(3, 64, 5)  # 8x8 grid
        out = pixel_shuffle(x, scale_factor=4)
        assert out.shape == (3, 4, 80)  # 64/16=4 tokens, 5*16=80 embed dim

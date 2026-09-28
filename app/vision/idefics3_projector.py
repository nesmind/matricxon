import torch
from torch import nn

from app.gguf.loader import GGUFModelLoader


def compute_num_patches(raw_num_patches: int, scale_factor: int) -> int:
    """The public patch count this projector produces per image.

    Idefics3's pixel-shuffle divides the raw (pre-shuffle) patch grid by
    `scale_factor**2` - this is the number `ClipVisionEncoder.num_patches`
    reports and `vision_fusion.py` allocates placeholder tokens for.
    """
    merge = scale_factor * scale_factor
    if raw_num_patches % merge != 0:
        raise ValueError(
            f"raw patch count {raw_num_patches} is not divisible by scale_factor**2={merge}"
        )
    return raw_num_patches // merge


def load_projector_weights(loader: GGUFModelLoader) -> tuple[torch.Tensor, int, int]:
    """Loads the real `mm.model.fc.weight` and derives (weight, projection_dim, scale_factor).

    Real, confirmed on ggml-org/SmolVLM-256M-Instruct-GGUF's own mmproj: a single bias-free
    linear (no "mm.model.fc.bias" tensor at all), and `clip.vision.projection_dim`'s metadata
    value does agree with this tensor's real shape here (unlike the LLaVA/MLP case) - still read
    from the tensor, not the metadata, to keep one consistent "shape wins" rule across both
    projector types.
    """
    weight = loader.load_tensor("mm.model.fc.weight")
    projection_dim = weight.shape[0]
    scale_factor = loader.metadata.require("clip.vision.projector.scale_factor")
    return weight, projection_dim, scale_factor


def build_projector(
    n_embd: int, scale_factor: int, projection_dim: int, dtype: torch.dtype
) -> nn.Linear:
    mm_fc_in = n_embd * scale_factor * scale_factor
    return nn.Linear(mm_fc_in, projection_dim, bias=False, dtype=dtype)


def pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Idefics3/SmolVLM's real projector pixel-shuffle patch merge.

    Extended into `ClipVisionEncoder` on 2026-09-28 against a real
    `ggml-org/SmolVLM-256M-Instruct-GGUF` mmproj pull -
    `general.base_model.1.name = "Siglip Base Patch16 512"`,
    `clip.projector_type = "idefics3"`. This spatially merges
    `scale_factor x scale_factor` adjacent patches into one (quadrupling the
    embedding dim, dividing the patch count by `scale_factor**2`), and is
    followed (in `ClipVisionEncoder.forward`) by a *single* bias-free linear
    projection (`mm.model.fc.weight` - real, confirmed: this file has no
    matching `.bias` tensor at all, unlike the MLP path's two-tensor
    `mm.0`/`mm.2`). The vision tower itself (SigLIP: no CLS token, no
    pre_ln, a real post_ln) needed zero code changes for this - it's the
    exact same tower shape moondream2 already validated.

    Deferred (real, substantial follow-up work, not part of this pass):
    Idefics3's own image-splitting preprocessing (a grid of sub-image tiles
    plus a global thumbnail, confirmed via the real
    `HuggingFaceTB/SmolVLM-256M-Instruct` preprocessor_config.json:
    `"do_image_splitting": true`) and its row/col separator prompt tokens -
    `ClipVisionEncoder` currently still encodes exactly one whole-image
    tensor per image, same as the MLP path, which is real working output
    but not a faithful reproduction of the reference model's full
    multi-tile pipeline.

    Ported directly from HF transformers' own
    `Idefics3Connector.pixel_shuffle`
    (`transformers/models/idefics3/modeling_idefics3.py`), not re-derived
    from ggml's `clip.cpp`/`siglip.cpp` C implementation
    (`build_patch_merge_permute`), whose reversed-axis ggml tensor
    convention would be easy to mistranslate into PyTorch. Assumes a square
    patch grid in row-major order - exactly what `ClipVisionEncoder.
    forward`'s own `x.flatten(2).transpose(1, 2)` (from a
    `(batch, n_embd, grid_h, grid_w)` conv output) already produces.
    """
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    merged = embed_dim * scale_factor * scale_factor
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, width // scale_factor, height // scale_factor, merged)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, seq // (scale_factor * scale_factor), merged)
    return x

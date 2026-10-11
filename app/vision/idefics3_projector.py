import torch
from torch import nn

from app.gguf.loader import GGUFModelLoader


def compute_num_patches(raw_num_patches: int, scale_factor: int) -> int:
    """The public patch count this projector produces per image. Idefics3's pixel-shuffle divides
    the raw (pre-shuffle) patch grid by `scale_factor**2`; this is the number
    `ClipVisionEncoder.num_patches` reports and `vision_fusion.py` allocates placeholder tokens
    for."""
    merge = scale_factor * scale_factor
    if raw_num_patches % merge != 0:
        raise ValueError(
            f"raw patch count {raw_num_patches} is not divisible by scale_factor**2={merge}"
        )
    return raw_num_patches // merge


def load_projector_weights(loader: GGUFModelLoader) -> tuple[torch.Tensor, int, int]:
    """Loads `mm.model.fc.weight` and derives (weight, projection_dim, scale_factor). On
    ggml-org/SmolVLM-256M-Instruct-GGUF's mmproj it is a single bias-free linear (no
    "mm.model.fc.bias") and `clip.vision.projection_dim` agrees with the tensor's shape (unlike the
    LLaVA/MLP case); still read from the tensor, keeping one "shape wins" rule across both projector
    types."""
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
    """Idefics3/SmolVLM's projector pixel-shuffle patch merge, added to `ClipVisionEncoder` against
    a `ggml-org/SmolVLM-256M-Instruct-GGUF` mmproj (`general.base_model.1.name = "Siglip Base
    Patch16 512"`, `clip.projector_type = "idefics3"`). It merges `scale_factor x scale_factor`
    adjacent patches into one (quadrupling the embedding dim, dividing the patch count by
    `scale_factor**2`), followed in `ClipVisionEncoder.forward` by a single bias-free linear
    (`mm.model.fc.weight`; no `.bias`, unlike the MLP path's `mm.0`/`mm.2`). The SigLIP tower (no
    CLS token, no pre_ln, a post_ln) needed no change: it is the shape moondream2 validated.

    Deferred: Idefics3's image splitting (a grid of sub-image tiles plus a global thumbnail;
    `HuggingFaceTB/SmolVLM-256M-Instruct`'s preprocessor_config.json has `"do_image_splitting":
    true`) and its row/col separator prompt tokens. `ClipVisionEncoder` still encodes one
    whole-image tensor per image, as the MLP path: working output, but not the reference model's
    multi-tile pipeline.

    Ported from HF transformers' `Idefics3Connector.pixel_shuffle`
    (`transformers/models/idefics3/modeling_idefics3.py`), not re-derived from ggml's
    `clip.cpp`/`siglip.cpp` (`build_patch_merge_permute`), whose reversed-axis convention is easy to
    mistranslate. Assumes a square patch grid in row-major order, which
    `ClipVisionEncoder.forward`'s `x.flatten(2).transpose(1, 2)` (from a `(batch, n_embd, grid_h,
    grid_w)` conv output) produces.
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

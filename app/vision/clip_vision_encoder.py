import torch
import torch.nn.functional as F
from torch import nn

from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.vision.clip_vision_encoder_layer import ClipVisionEncoderLayer
from app.vision.idefics3_projector import (
    build_projector,
    compute_num_patches,
    load_projector_weights,
    pixel_shuffle,
)

_PROJECTOR_MLP = "mlp"
_PROJECTOR_IDEFICS3 = "idefics3"


class ClipVisionEncoder(nn.Module):
    """The real `clip`-architecture vision tower + projector llama.cpp's

    `clip.cpp`/mmproj GGUF format packages. Validated against three real
    mmproj pulls, each conditional on real tensor presence so earlier paths
    stay unchanged:
    - `moondream/moondream2-gguf`: SigLIP-shaped (no CLS token, no pre_ln, a
      real post_ln) + a 2-layer MLP projector (`mm.0`/`mm.2`,
      `clip.projector_type = "mlp"`).
    - `second-state/Llava-v1.6-Vicuna-7B-GGUF` (2026-09-21): OpenAI-CLIP-
      shaped towers add a CLS token (`v.class_embd`, dropped again before
      the projector), a pre-transformer LayerNorm (`v.pre_ln`), and may omit
      `post_ln` entirely (skipped when the tensor is absent).
    - `ggml-org/SmolVLM-256M-Instruct-GGUF` (2026-09-28): Idefics3's
      `idefics3` projector (pixel-shuffle merge + a single bias-free linear)
      - see `app.vision.idefics3_projector.pixel_shuffle`'s own docstring
      for the full history and real-file verification detail.

    Still deliberately **not** a `ModelArchitecture` subclass / not
    registered in `ArchitectureRegistry` - this stays a standalone component
    `chat_router.py` invokes directly for a paired mmproj (see
    `ModelCatalog.find_paired_mmproj`), not part of `ModelManager`'s own
    load/evict lifecycle.
    """

    def __init__(
        self,
        metadata: GGUFMetadata,
        projection_dim: int,
        has_class_embd: bool,
        has_pre_ln: bool,
        has_post_ln: bool,
        has_patch_embd_bias: bool = True,
        projector_type: str = _PROJECTOR_MLP,
        projector_hidden: int | None = None,
        scale_factor: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_embd = metadata.get_u32("clip.vision.embedding_length")
        self.n_head = metadata.get_u32("clip.vision.attention.head_count")
        self.n_layer = metadata.get_u32("clip.vision.block_count")
        self.ffn_len = metadata.get_u32("clip.vision.feed_forward_length")
        self.eps = metadata.get_f32("clip.vision.attention.layer_norm_epsilon")
        self.patch_size = metadata.get_u32("clip.vision.patch_size")
        self.image_size = metadata.get_u32("clip.vision.image_size")
        self.projection_dim = projection_dim
        self.image_mean: list[float] = metadata.require("clip.vision.image_mean")
        self.image_std: list[float] = metadata.require("clip.vision.image_std")
        self.has_class_embd = has_class_embd
        self.projector_type = projector_type
        self.scale_factor = scale_factor

        # Position embeddings apply at the raw (pre-pixel-shuffle) patch resolution.
        self.raw_num_patches = (self.image_size // self.patch_size) ** 2
        if projector_type == _PROJECTOR_IDEFICS3:
            self.num_patches = compute_num_patches(self.raw_num_patches, scale_factor)
        else:
            self.num_patches = self.raw_num_patches

        self.patch_embd = nn.Conv2d(
            3,
            self.n_embd,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=has_patch_embd_bias,
            dtype=dtype,
        )
        self.class_embd = (
            nn.Parameter(torch.zeros(self.n_embd, dtype=dtype)) if has_class_embd else None
        )
        position_len = self.raw_num_patches + 1 if has_class_embd else self.raw_num_patches
        self.position_embd = nn.Parameter(torch.zeros(position_len, self.n_embd, dtype=dtype))
        self.pre_ln = nn.LayerNorm(self.n_embd, eps=self.eps, dtype=dtype) if has_pre_ln else None
        self.layers = nn.ModuleList(
            [
                ClipVisionEncoderLayer(
                    self.n_embd, self.n_head, self.ffn_len, self.eps, dtype=dtype
                )
                for _ in range(self.n_layer)
            ]
        )
        self.post_ln = (
            nn.LayerNorm(self.n_embd, eps=self.eps, dtype=dtype) if has_post_ln else None
        )

        self.projector_up: nn.Linear | None = None
        self.projector_down: nn.Linear | None = None
        self.mm_fc: nn.Linear | None = None
        if projector_type == _PROJECTOR_IDEFICS3:
            self.mm_fc = build_projector(self.n_embd, scale_factor, self.projection_dim, dtype)
        elif projector_type == _PROJECTOR_MLP:
            if projector_hidden is None:
                raise ValueError("projector_hidden is required for the mlp projector type")
            self.projector_up = nn.Linear(self.n_embd, projector_hidden, dtype=dtype)
            self.projector_down = nn.Linear(projector_hidden, self.projection_dim, dtype=dtype)
        else:
            raise ValueError(f"Unsupported clip.projector_type: {projector_type!r}")

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == "clip" and metadata.get_bool(
            "clip.has_vision_encoder", False
        )

    @classmethod
    def from_gguf(
        cls, loader: GGUFModelLoader, dtype: torch.dtype = torch.float32
    ) -> "ClipVisionEncoder":
        """Eager loading - unlike the text decoders' on-the-fly dequant, this

        component isn't part of `ModelManager`'s load/evict lifecycle, so
        there's no latency/memory motivation to defer it.
        """
        projector_type = loader.metadata.get_str("clip.projector_type", _PROJECTOR_MLP)
        has_class_embd = loader.has_tensor("v.class_embd")
        has_pre_ln = loader.has_tensor("v.pre_ln.weight")
        has_post_ln = loader.has_tensor("v.post_ln.weight")
        has_patch_embd_bias = loader.has_tensor("v.patch_embd.bias")

        projector_up_weight: torch.Tensor | None = None
        projector_hidden: int | None = None
        idefics3_fc_weight: torch.Tensor | None = None
        scale_factor = 1
        if projector_type == _PROJECTOR_IDEFICS3:
            idefics3_fc_weight, projection_dim, scale_factor = load_projector_weights(loader)
        else:
            # mm.0/mm.2's real shapes win over metadata: the MLP hidden width has no `clip.*`
            # metadata key at all, and `clip.vision.projection_dim` is confirmed *wrong* on the
            # real LLaVA mmproj file (768 = CLIP's native contrastive-projection width, not this
            # file's real mm-projector output of 4096) - trusting it would crash on `.copy_()`.
            projector_up_weight = loader.load_tensor("mm.0.weight")
            projector_hidden = projector_up_weight.shape[0]
            projection_dim = loader.load_tensor("mm.2.weight").shape[0]

        model = cls(
            loader.metadata,
            projection_dim,
            has_class_embd,
            has_pre_ln,
            has_post_ln,
            has_patch_embd_bias=has_patch_embd_bias,
            projector_type=projector_type,
            projector_hidden=projector_hidden,
            scale_factor=scale_factor,
            dtype=dtype,
        )
        with torch.no_grad():
            model.patch_embd.weight.copy_(loader.load_tensor("v.patch_embd.weight"))
            if model.patch_embd.bias is not None:
                model.patch_embd.bias.copy_(loader.load_tensor("v.patch_embd.bias"))
            if model.class_embd is not None:
                model.class_embd.copy_(loader.load_tensor("v.class_embd"))
            position_len = model.raw_num_patches + 1 if has_class_embd else model.raw_num_patches
            model.position_embd.copy_(
                loader.load_tensor("v.position_embd.weight").reshape(position_len, model.n_embd)
            )
            if model.pre_ln is not None:
                model.pre_ln.weight.copy_(loader.load_tensor("v.pre_ln.weight"))
                model.pre_ln.bias.copy_(loader.load_tensor("v.pre_ln.bias"))
            for i, layer in enumerate(model.layers):
                prefix = f"v.blk.{i}."
                layer.ln1.weight.copy_(loader.load_tensor(prefix + "ln1.weight"))
                layer.ln1.bias.copy_(loader.load_tensor(prefix + "ln1.bias"))
                layer.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
                layer.q_proj.bias.copy_(loader.load_tensor(prefix + "attn_q.bias"))
                layer.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
                layer.k_proj.bias.copy_(loader.load_tensor(prefix + "attn_k.bias"))
                layer.v_proj.weight.copy_(loader.load_tensor(prefix + "attn_v.weight"))
                layer.v_proj.bias.copy_(loader.load_tensor(prefix + "attn_v.bias"))
                layer.out_proj.weight.copy_(loader.load_tensor(prefix + "attn_out.weight"))
                layer.out_proj.bias.copy_(loader.load_tensor(prefix + "attn_out.bias"))
                layer.ln2.weight.copy_(loader.load_tensor(prefix + "ln2.weight"))
                layer.ln2.bias.copy_(loader.load_tensor(prefix + "ln2.bias"))
                # Real, confirmed-by-bias-length naming swap (see ClipVisionEncoderLayer's own
                # docstring): the tensor named "ffn_down" is actually the expanding projection
                # (fc1, n_embd -> ffn_len) and "ffn_up" is the contracting one (fc2).
                layer.fc1.weight.copy_(loader.load_tensor(prefix + "ffn_down.weight"))
                layer.fc1.bias.copy_(loader.load_tensor(prefix + "ffn_down.bias"))
                layer.fc2.weight.copy_(loader.load_tensor(prefix + "ffn_up.weight"))
                layer.fc2.bias.copy_(loader.load_tensor(prefix + "ffn_up.bias"))
            if model.post_ln is not None:
                model.post_ln.weight.copy_(loader.load_tensor("v.post_ln.weight"))
                model.post_ln.bias.copy_(loader.load_tensor("v.post_ln.bias"))
            if model.mm_fc is not None:
                model.mm_fc.weight.copy_(idefics3_fc_weight)
            else:
                model.projector_up.weight.copy_(projector_up_weight)
                model.projector_up.bias.copy_(loader.load_tensor("mm.0.bias"))
                model.projector_down.weight.copy_(loader.load_tensor("mm.2.weight"))
                model.projector_down.bias.copy_(loader.load_tensor("mm.2.bias"))
        return model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (batch, 3, image_size, image_size) -> projected

        embeddings (batch, num_patches, projection_dim), one soft token per
        real image patch (see `ChatRequestHandler` for the splice into a
        chat request's input embeddings). A CLS token, when present, is used
        internally but dropped before the projector.
        """
        x = self.patch_embd(pixel_values)  # (batch, n_embd, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, n_embd)
        if self.class_embd is not None:
            batch = x.shape[0]
            cls = self.class_embd.expand(batch, 1, self.n_embd)
            x = torch.cat([cls, x], dim=1)  # (batch, 1 + num_patches, n_embd)
        x = x + self.position_embd

        if self.pre_ln is not None:
            x = self.pre_ln(x)

        for layer in self.layers:
            x = layer(x)
        if self.post_ln is not None:
            x = self.post_ln(x)

        if self.class_embd is not None:
            x = x[:, 1:, :]  # drop the CLS position - see this method's own docstring

        if self.mm_fc is not None:
            x = pixel_shuffle(x, self.scale_factor)
            x = self.mm_fc(x)
        else:
            x = self.projector_down(F.gelu(self.projector_up(x)))
        return x

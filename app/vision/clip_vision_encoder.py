import torch
import torch.nn.functional as F
from torch import nn

from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata


class ClipVisionEncoderLayer(nn.Module):
    """A standard pre-norm ViT block - plain (non-causal, non-grouped)

    multi-head self-attention with real biases on every projection
    (confirmed via this GGUF's real tensor list, unlike matricxon's
    bias-free text decoders), then a 2-layer GELU MLP. Pre-norm (LN before
    each sub-block), not `BertArchitecture`'s post-norm - confirmed by the
    real tensor names (`ln1`/`ln2` feed into, not out of, their sub-block).

    `fc1`/`fc2` (expand then contract) are named generically on purpose,
    not `ffn_up`/`ffn_down` - a real, confirmed surprise (via the tensors'
    own bias lengths, not assumed from the names) is that this GGUF's
    actual `ffn_up.weight`/`ffn_up.bias` is the *contracting* projection
    (bias has `n_embd` elements) and `ffn_down` is the *expanding* one
    (bias has `ffn_len` elements) - the opposite of what those names would
    suggest. See `ClipVisionEncoder.from_gguf`'s loading code for exactly
    which real tensor feeds which of `fc1`/`fc2`.
    """

    def __init__(
        self, n_embd: int, n_head: int, ffn_len: int, eps: float, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd, eps=eps, dtype=dtype)
        self.q_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.out_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.ln2 = nn.LayerNorm(n_embd, eps=eps, dtype=dtype)
        self.fc1 = nn.Linear(n_embd, ffn_len, dtype=dtype)
        self.fc2 = nn.Linear(ffn_len, n_embd, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape

        residual = x
        h = self.ln1(x)
        q = self.q_proj(h).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v
        )  # no causal mask - a full image, not a sequence
        attn = attn.transpose(1, 2).reshape(batch, seq_len, n_embd)
        x = residual + self.out_proj(attn)

        residual = x
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return residual + h


class ClipVisionEncoder(nn.Module):
    """The real `clip`-architecture vision tower + MLP projector

    llama.cpp's `clip.cpp`/mmproj GGUF format packages. Originally built and
    validated against a real `moondream/moondream2-gguf` mmproj pull (a
    SigLIP-shaped ViT: learned absolute position embeddings, no CLS token,
    no pre-transformer LayerNorm, a final `post_ln` applied to the whole
    patch sequence) followed by a real 2-layer MLP projector (`mm.0`/`mm.2`,
    `clip.projector_type = "mlp"`) into the paired text model's embedding
    space.

    Extended (2026-09-21, real `second-state/Llava-v1.6-Vicuna-7B-GGUF`
    mmproj pull) to also handle real OpenAI-CLIP-shaped vision towers, which
    differ in three confirmed, real ways from SigLIP - each made
    conditional on the corresponding tensor's real presence, so the
    already-validated moondream2/SigLIP path (none of the three tensors
    exist there) is completely unchanged:
    - a real CLS token (`v.class_embd`, prepended before the patch sequence;
      `v.position_embd.weight`'s real shape is `num_patches + 1` here, not
      `num_patches`) - dropped again before the projector, since LLaVA's
      own real usage feeds only patch features forward, not the CLS token
      (the one design choice here not directly read off metadata - flagged
      for empirical verification against real output, not certain);
    - a real pre-transformer LayerNorm (`v.pre_ln`), applied once right
      after the position embedding add, before the first encoder layer;
    - **no** final `post_ln` at all on this file (confirmed: `v.post_ln.*`
      simply isn't in this GGUF's real tensor list - the original code
      loaded it unconditionally, which would have crashed outright on this
      file) - skipped (identity) when absent rather than assumed present.

    Still deliberately **not** a `ModelArchitecture` subclass / not
    registered in `ArchitectureRegistry` - this stays a standalone component
    `chat_router.py` invokes directly for a paired mmproj (see
    `ModelCatalog.find_paired_mmproj`), not part of `ModelManager`'s own
    load/evict lifecycle.
    """

    def __init__(
        self,
        metadata: GGUFMetadata,
        projector_hidden: int,
        projection_dim: int,
        has_class_embd: bool,
        has_pre_ln: bool,
        has_post_ln: bool,
        has_patch_embd_bias: bool = True,
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
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.has_class_embd = has_class_embd

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
        position_len = self.num_patches + 1 if has_class_embd else self.num_patches
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
        self.projector_up = nn.Linear(self.n_embd, projector_hidden, dtype=dtype)
        self.projector_down = nn.Linear(projector_hidden, self.projection_dim, dtype=dtype)

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == "clip" and metadata.get_bool(
            "clip.has_vision_encoder", False
        )

    @classmethod
    def from_gguf(
        cls, loader: GGUFModelLoader, dtype: torch.dtype = torch.float32
    ) -> "ClipVisionEncoder":
        """Eager loading (unlike the text decoders' on-the-fly dequant) -

        this component isn't part of `ModelManager`'s load/evict lifecycle
        at all (see this class's own docstring on why), so there's no
        request-latency or memory-pressure motivation to defer it.
        """
        # The MLP projector's hidden width isn't exposed via any `clip.*` metadata key (confirmed
        # on the real moondream2 mmproj GGUF) - only derivable from `mm.0.weight`'s own real shape
        # (1152->8192->2048 on that file), so it's read here before construction rather than
        # hardcoded, letting a differently-sized real (or tiny synthetic test) projector still work.
        projector_up_weight = loader.load_tensor("mm.0.weight")
        projector_hidden = projector_up_weight.shape[0]
        # Same reasoning, extended: `clip.vision.projection_dim` is confirmed *wrong* on the real
        # LLaVA mmproj file (metadata says 768 - CLIP's own native contrastive-projection width,
        # unrelated to this file's real mm-projector output - while `mm.2.weight`'s real shape is
        # 4096, matching the paired Vicuna-7B text model's actual hidden size). Trusting the
        # metadata here would silently build a projector with the wrong output width and crash on
        # the very next `.copy_()` - real tensor shape wins, same as `projector_hidden` above.
        projection_dim = loader.load_tensor("mm.2.weight").shape[0]
        has_class_embd = loader.has_tensor("v.class_embd")
        has_pre_ln = loader.has_tensor("v.pre_ln.weight")
        has_post_ln = loader.has_tensor("v.post_ln.weight")
        has_patch_embd_bias = loader.has_tensor("v.patch_embd.bias")

        model = cls(
            loader.metadata,
            projector_hidden,
            projection_dim,
            has_class_embd,
            has_pre_ln,
            has_post_ln,
            has_patch_embd_bias=has_patch_embd_bias,
            dtype=dtype,
        )
        with torch.no_grad():
            model.projector_up.weight.copy_(projector_up_weight)
            model.patch_embd.weight.copy_(loader.load_tensor("v.patch_embd.weight"))
            if model.patch_embd.bias is not None:
                model.patch_embd.bias.copy_(loader.load_tensor("v.patch_embd.bias"))
            if model.class_embd is not None:
                model.class_embd.copy_(loader.load_tensor("v.class_embd"))
            position_len = model.num_patches + 1 if has_class_embd else model.num_patches
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
            model.projector_up.bias.copy_(loader.load_tensor("mm.0.bias"))
            model.projector_down.weight.copy_(loader.load_tensor("mm.2.weight"))
            model.projector_down.bias.copy_(loader.load_tensor("mm.2.bias"))
        return model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (batch, 3, image_size, image_size) -> projected

        embeddings (batch, num_patches, projection_dim) - one "soft token"
        per real image patch, in the paired text model's embedding space
        (see `app.routers.chat_router.ChatRequestHandler` for where these
        get spliced into a real chat request's input embeddings). The CLS
        token, when this file has one, is used internally (it's a real part
        of the pre-trained position/attention structure) but dropped from
        the returned sequence - LLaVA's own real usage feeds only patch
        features onward, never the CLS token itself.
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

        x = self.projector_down(F.gelu(self.projector_up(x)))
        return x

import torch
import torch.nn.functional as F
from torch import nn

from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata

_ROPE_THETA = 10000.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Qwen3VLVisionLayer(nn.Module):
    """Pre-LN ViT block: fused-qkv full attention with 2D rotary q/k, then a tanh-GELU MLP."""

    def __init__(self, n_embd: int, n_head: int, ffn_len: int, eps: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(n_embd, eps=eps)
        self.ln2 = nn.LayerNorm(n_embd, eps=eps)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.out = nn.Linear(n_embd, n_embd)
        self.fc1 = nn.Linear(n_embd, ffn_len)
        self.fc2 = nn.Linear(ffn_len, n_embd)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = self.qkv(self.ln1(x)).view(n, 3, self.n_head, -1).unbind(dim=1)
        q = q * cos + _rotate_half(q) * sin  # cos/sin: (n, 1, head_dim)
        k = k * cos + _rotate_half(k) * sin
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # (1, heads, n, d)
        attn = F.scaled_dot_product_attention(q, k, v)[0].transpose(0, 1).reshape(n, -1)
        x = x + self.out(attn)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class Qwen3VLVisionEncoder(nn.Module):
    """The `qwen3vl_merger` vision tower + projector of a Qwen3-VL / Qwen3.5 mmproj GGUF
    (`clip.projector_type`), confirmed against the real `unsloth/Qwen3.5-4B-GGUF` mmproj header
    (2026-10-01) and HF's `Qwen3VLVisionModel`:

    - patch embedding is a Conv3d with a 2-frame temporal kernel, stored as two Conv2d kernels
      (`v.patch_embd.weight` / `.weight.1`); an image is two identical frames, so the two kernels
      simply sum;
    - a learned `grid x grid` position table, bilinearly resized (align_corners) to the image's
      patch grid;
    - 2D rotary attention (row/column frequencies, NEOX rotate-half over the head dim), full
      attention over every patch of the image;
    - `v.post_ln` then a 2x2 spatial merge (4 neighbouring patches -> one 4*n_embd vector) and a
      2-layer exact-GELU MLP (`mm.0`/`mm.2`) to the text model's width. No deepstack layers in
      the real file (`is_deepstack_layers` is all False, no `v.deepstack.*` tensors).

    Patches stay in plain raster order throughout (HF reorders them into merge windows first, but
    attention is global per image and rotary/position are per-patch, so only the final merge
    grouping depends on order). Output: one row per 2x2 merged token, raster order over the
    merged grid.
    """

    def __init__(self, metadata: GGUFMetadata, out_dim: int, mlp_hidden: int) -> None:
        super().__init__()
        self.n_embd = metadata.get_u32("clip.vision.embedding_length")
        self.n_head = metadata.get_u32("clip.vision.attention.head_count")
        n_layer = metadata.get_u32("clip.vision.block_count")
        ffn_len = metadata.get_u32("clip.vision.feed_forward_length")
        eps = metadata.get_f32("clip.vision.attention.layer_norm_epsilon")
        self.patch_size = metadata.get_u32("clip.vision.patch_size")
        self.merge = metadata.get_u32("clip.vision.spatial_merge_size")
        self.grid = metadata.get_u32("clip.vision.image_size") // self.patch_size
        self.image_mean: list[float] = metadata.require("clip.vision.image_mean")
        self.image_std: list[float] = metadata.require("clip.vision.image_std")
        self.out_dim = out_dim
        self.patch_embd = nn.Conv2d(3, self.n_embd, self.patch_size, self.patch_size)
        self.position_embd = nn.Parameter(torch.zeros(self.grid * self.grid, self.n_embd))
        self.layers = nn.ModuleList(
            [Qwen3VLVisionLayer(self.n_embd, self.n_head, ffn_len, eps) for _ in range(n_layer)]
        )
        self.post_ln = nn.LayerNorm(self.n_embd, eps=eps)
        merged = self.n_embd * self.merge * self.merge
        self.mm0 = nn.Linear(merged, mlp_hidden)
        self.mm2 = nn.Linear(mlp_hidden, out_dim)

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == "clip" and (
            metadata.get_str("clip.projector_type", "") == "qwen3vl_merger"
        )

    @classmethod
    def from_gguf(cls, loader: GGUFModelLoader) -> "Qwen3VLVisionEncoder":
        mm0, mm2 = loader.load_tensor("mm.0.weight"), loader.load_tensor("mm.2.weight")
        model = cls(loader.metadata, out_dim=mm2.shape[0], mlp_hidden=mm0.shape[0])
        load = loader.load_tensor
        with torch.no_grad():
            model.patch_embd.weight.copy_(
                load("v.patch_embd.weight") + load("v.patch_embd.weight.1")
            )
            model.patch_embd.bias.copy_(load("v.patch_embd.bias"))
            model.position_embd.copy_(load("v.position_embd.weight"))
            for i, layer in enumerate(model.layers):
                p = f"v.blk.{i}."
                for mod, name in (
                    (layer.ln1, "ln1"), (layer.ln2, "ln2"), (layer.qkv, "attn_qkv"),
                    (layer.out, "attn_out"), (layer.fc1, "ffn_up"), (layer.fc2, "ffn_down"),
                ):  # fmt: skip
                    mod.weight.copy_(load(p + name + ".weight"))
                    mod.bias.copy_(load(p + name + ".bias"))
            model.post_ln.weight.copy_(load("v.post_ln.weight"))
            model.post_ln.bias.copy_(load("v.post_ln.bias"))
            model.mm0.weight.copy_(mm0)
            model.mm0.bias.copy_(load("mm.0.bias"))
            model.mm2.weight.copy_(mm2)
            model.mm2.bias.copy_(load("mm.2.bias"))
        return model.eval()

    def merged_grid(self, height: int, width: int) -> tuple[int, int]:
        unit = self.patch_size * self.merge
        return height // unit, width // unit

    def _rotary(self, gh: int, gw: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin (n_patches, 1, head_dim): [row freqs | col freqs] duplicated for rotate-half."""
        head_dim = self.n_embd // self.n_head
        inv_freq = 1.0 / (
            _ROPE_THETA
            ** (torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2))
        )
        rows = torch.arange(gh).repeat_interleave(gw).float()
        cols = torch.arange(gw).repeat(gh).float()
        freqs = torch.cat([torch.outer(rows, inv_freq), torch.outer(cols, inv_freq)], dim=-1)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (1, 3, H, W), H/W multiples of patch_size * merge -> (tokens, out_dim)."""
        x = self.patch_embd(pixel_values)  # (1, n_embd, gh, gw)
        gh, gw = x.shape[-2:]
        x = x[0].flatten(1).transpose(0, 1)  # (gh*gw, n_embd), raster order
        table = self.position_embd.view(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(table, size=(gh, gw), mode="bilinear", align_corners=True)
        x = x + pos[0].flatten(1).transpose(0, 1)
        cos, sin = self._rotary(gh, gw)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.post_ln(x)
        m = self.merge
        x = (
            x.view(gh // m, m, gw // m, m, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, m * m * x.shape[-1])
        )
        return self.mm2(F.gelu(self.mm0(x)))

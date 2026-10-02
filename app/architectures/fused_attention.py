import torch

from app.native.fused_ops import FusedOps


def fused_cached_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_offset: object,
    scale: float | None = None,
    window: int | None = None,
) -> torch.Tensor | None:
    """Attention of a few new query tokens over the KV cache in one native call (`FusedOps`), or
    None when that doesn't apply (fused ops off, a prefill, a batched decode, non-float32...) and
    the caller runs its torch `scaled_dot_product_attention`. `scale=None` is the default
    1/sqrt(head_dim); `window` is a sliding-window size (None/0: full causal)."""
    ops = FusedOps.active()
    if ops is None:
        return None
    return ops.attention(
        q, k, v, cache_offset, q.shape[-1] ** -0.5 if scale is None else scale, window or 0
    )

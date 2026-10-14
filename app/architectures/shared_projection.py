"""Projections that read the same input (q/k/v, gate/up) as one native call during decode
(`MultiGemmPlan`): one input quantization and one thread fork instead of one per weight. Anything
else (prefill, Numba or torch weights, a bias-free mix) runs each projection as before."""

import torch
from torch import nn

from app.architectures.quantized_linear import QuantizedLinear
from app.native.gemm_multi import MultiGemmPlan

# Up to this many tokens per call (a decode step or a small decode batch).
MAX_SHARED_TOKENS = 8


def _plan(owner: nn.Module, names: tuple[str, ...], mods: list[nn.Module]) -> MultiGemmPlan | None:
    """Built once per layer and group, cached on the layer; rebuilt if its modules were replaced."""
    key = f"_mx_plan_{'_'.join(names)}"
    ids = tuple(id(m) for m in mods)
    cached = owner.__dict__.get(key)
    if cached is not None and cached[0] == ids:
        return cached[1]
    plan = None
    if all(isinstance(m, QuantizedLinear) and m.native_ready for m in mods):
        in_features = {m.in_features for m in mods}
        if len(in_features) == 1:
            plan = MultiGemmPlan.build(
                [(m.ggml_type, m.raw, m.out_features) for m in mods], in_features.pop()
            )
    owner.__dict__[key] = (ids, plan)
    return plan


def project_shared(owner: nn.Module, names: tuple[str, ...], x: torch.Tensor) -> list[torch.Tensor]:
    """`owner.<name>(x)` for each name, as one native call when possible."""
    mods = [getattr(owner, name) for name in names]
    batch, seq_len, in_features = x.shape
    if batch * seq_len <= MAX_SHARED_TOKENS and x.device.type == "cpu":
        plan = _plan(owner, names, mods)
        if plan is not None:
            ys = plan.run(x.reshape(-1, in_features))
            if ys is not None:
                return [m.finish(y, batch, seq_len) for m, y in zip(mods, ys, strict=True)]
    return [m(x) for m in mods]

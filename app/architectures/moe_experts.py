"""A shared, quantized-native-aware Mixture-of-Experts expert FFN - real, reusable core every
real MoE architecture in this project now composes (`GraniteMoeFFN`/Mixtral via
`granitemoe_layers.py`, `gemma4_moe.py`), instead of each hand-rolling its own sparse dispatch
loop over a real 3D `(num_experts, ffn_dim, hidden_size)` weight tensor. A new MoE architecture
only needs its own router (the real math genuinely differs per architecture - see
`gemma4_moe.Gemma4Router`'s own docstring for a documented example) and its own activation
function; the expert computation itself (gather routed tokens, apply that expert's own gate/up/
down weights, scale, scatter back) is exactly this class, unchanged.

Closes a real, confirmed gap (2026-09-29, see ROADMAP.md's own entry this replaces): every MoE
architecture previously always fully dequantized its expert tensors at load time -
`_load_projection` (the machinery behind `QuantizedLinear`/the native-C and Numba GEMV kernels)
hard-assumes a plain 2D `(out_features, in_features)` weight, which a real 3D expert tensor
isn't. The fix doesn't need new kernels at all: a real per-expert tensor is a contiguous, real 2D
slice of the bigger 3D one - `_materialize_one` below just slices the right byte range per expert
and hands it to the exact same, already-real, already-tested `QuantizedLinear` unchanged.

The big dense `nn.Parameter` fallback (today's exact pre-existing behavior) is always allocated
at construction time, same as every 2D projection's own placeholder `nn.Linear` already is - but
once a projection actually packs, `_materialize_one` frees it (reassigns the attribute to a
0-sized stub) rather than leaving a full, permanently-unused dense copy allocated forever
alongside the packed one - the one real way this differs from the 2D placeholder-then-replace
precedent, and the one that matters most here: expert tensors are usually a real MoE checkpoint's
single biggest chunk of memory.
"""

from collections.abc import Callable

import torch
from torch import nn

from app.architectures.device_packed import DevicePackedLinear, upload_packed
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.dequant.quantized_gemv_registry import has_gemv_kernel
from app.gguf.dequant.torch_dequant import TorchDequantizer
from app.gguf.loader import GGUFModelLoader


class QuantizedMoEExperts(nn.Module):
    def __init__(
        self,
        n_embd: int,
        ffn_len: int,
        num_experts: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.activation = activation
        self.dtype = dtype
        self.gate_exps = nn.Parameter(torch.empty(num_experts, ffn_len, n_embd, dtype=dtype))
        self.up_exps = nn.Parameter(torch.empty(num_experts, ffn_len, n_embd, dtype=dtype))
        self.down_exps = nn.Parameter(torch.empty(num_experts, n_embd, ffn_len, dtype=dtype))
        # Packed (quantized-native) per-expert QuantizedLinear, keyed by str(expert_id) - real
        # nn.Module children (nn.ModuleDict requires string keys), so
        # ModelArchitecture._release_packed_modules's own `self.modules()` walk already finds and
        # releases every one of these automatically before its mmap closes - no changes needed
        # there at all.
        self.gate_packed = nn.ModuleDict()
        self.up_packed = nn.ModuleDict()
        self.down_packed = nn.ModuleDict()
        # Set by `DevicePlacement.place_on` in packed-GPU mode: experts then upload still packed.
        self.pack_device: torch.device | None = None

    def deferred_params(self) -> list[nn.Parameter]:
        """The big dense placeholders `place_on` must not allocate on the device up front."""
        return [self.gate_exps, self.up_exps, self.down_exps]

    def use_device_packing(self, device: torch.device) -> None:
        self.pack_device = device

    def _project(
        self, packed: nn.ModuleDict, dense: nn.Parameter, expert_id: int, x: torch.Tensor
    ) -> torch.Tensor:
        key = str(expert_id)
        if key in packed:
            return packed[key](x.unsqueeze(0)).squeeze(0)
        return x @ dense[expert_id].T

    def forward(
        self, x: torch.Tensor, top_k_weights: torch.Tensor, top_k_idx: torch.Tensor
    ) -> torch.Tensor:
        """`x`: `(n_tokens, n_embd)`; `top_k_weights`/`top_k_idx`: `(n_tokens, num_experts_per_tok)`
        - the router's own output, in whatever real order/normalization that architecture's own
        router produces (this class doesn't care, it only ever reads the values it's handed)."""
        n_tokens, n_embd = x.shape
        out = torch.zeros(n_tokens, n_embd, dtype=torch.float32, device=x.device)
        for expert_id in top_k_idx.unique().tolist():
            token_idx, k_idx = (top_k_idx == expert_id).nonzero(as_tuple=True)
            x_e = x.index_select(0, token_idx)
            gate = self.activation(self._project(self.gate_packed, self.gate_exps, expert_id, x_e))
            up = self._project(self.up_packed, self.up_exps, expert_id, x_e)
            down = self._project(self.down_packed, self.down_exps, expert_id, gate * up)
            weight = top_k_weights[token_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, (down * weight).to(torch.float32))
        return out.to(x.dtype)


def _materialize_one(
    experts: QuantizedMoEExperts,
    dense_attr: str,
    packed_attr: str,
    loader: GGUFModelLoader,
    tensor_name: str,
    enabled: bool,
    dtype: torch.dtype,
) -> bool:
    """Packs every real expert into its own `QuantizedLinear` when `enabled` and a real GEMV
    kernel exists for this tensor's type, freeing the big dense placeholder once every expert is
    packed (see this module's own docstring); otherwise falls straight through to today's exact
    eager `.copy_()`. Returns whether it packed (the caller only needs to keep this loader's mmap
    open past materialization - see `_mark_quantized_native_used` - if at least one real tensor
    actually did)."""
    if enabled and experts.pack_device is not None:
        raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
        if TorchDequantizer.supports(ggml_type):
            num_experts, out_features, in_features = shape
            per_expert_bytes = len(raw) // num_experts
            packed = getattr(experts, packed_attr)
            for expert_id in range(num_experts):
                start = expert_id * per_expert_bytes
                rows = upload_packed(
                    raw[start : start + per_expert_bytes], out_features, experts.pack_device
                )
                packed[str(expert_id)] = DevicePackedLinear(
                    rows, in_features, ggml_type, None, dtype
                )
            setattr(experts, dense_attr, nn.Parameter(torch.empty(0, dtype=dtype)))
            return False  # bytes were copied to the device: the mmap need not stay open
    elif enabled:
        raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
        if has_gemv_kernel(ggml_type):
            num_experts, out_features, in_features = shape
            per_expert_bytes = len(raw) // num_experts
            packed = getattr(experts, packed_attr)
            for expert_id in range(num_experts):
                start = expert_id * per_expert_bytes
                expert_raw = raw[start : start + per_expert_bytes]
                packed[str(expert_id)] = QuantizedLinear(
                    out_features, in_features, ggml_type, expert_raw, dtype=dtype
                )
            setattr(experts, dense_attr, nn.Parameter(torch.empty(0, dtype=dtype)))
            return True
    getattr(experts, dense_attr).data.copy_(loader.load_tensor(tensor_name))
    return False


def materialize_quantized_moe_experts(
    experts: QuantizedMoEExperts,
    loader: GGUFModelLoader,
    prefix: str,
    enabled: bool,
    mark_used: Callable[[GGUFModelLoader], None],
    gate_name: str = "ffn_gate_exps.weight",
    up_name: str = "ffn_up_exps.weight",
    down_name: str = "ffn_down_exps.weight",
) -> None:
    packed_any = False
    packed_any |= _materialize_one(
        experts, "gate_exps", "gate_packed", loader, prefix + gate_name, enabled, experts.dtype
    )
    packed_any |= _materialize_one(
        experts, "up_exps", "up_packed", loader, prefix + up_name, enabled, experts.dtype
    )
    packed_any |= _materialize_one(
        experts, "down_exps", "down_packed", loader, prefix + down_name, enabled, experts.dtype
    )
    if packed_any:
        mark_used(loader)

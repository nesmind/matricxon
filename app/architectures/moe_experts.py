"""A shared, quantized-native-aware MoE expert FFN, the reusable core of every MoE architecture here
(`GraniteMoeFFN`/Mixtral via `granitemoe_layers.py`, `gemma4_moe.py`) instead of a hand-rolled
sparse dispatch loop over a 3D `(num_experts, ffn_dim, hidden_size)` weight. A new MoE architecture
needs only its router (math differs; see `gemma4_moe.Gemma4Router`) and activation; this class does
the expert work (gather routed tokens, apply gate/up/down, scale, scatter back).

Closes a gap: MoE architectures dequantized expert tensors at load, since
`_load_projection` (behind `QuantizedLinear` and the native/Numba GEMV kernels) assumes a 2D weight.
No new kernels: a per-expert tensor is a contiguous 2D slice of the 3D one, so `_materialize_one`
slices the byte range per expert for the existing `QuantizedLinear`.

The dense `nn.Parameter` fallback is allocated at construction, like each 2D projection's
placeholder; once a projection packs, `_materialize_one` frees it (a 0-sized stub) instead of
keeping an unused dense copy, which matters since expert tensors are usually a MoE checkpoint's
biggest memory chunk.
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
        # Packed per-expert QuantizedLinear, keyed by str(expert_id) (nn.ModuleDict needs string
        # keys): real nn.Module
        # children, so ModelArchitecture._release_packed_modules's `self.modules` walk finds and
        # releases them before the mmap closes.
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
        """`x`: `(n_tokens, n_embd)`; `top_k_weights`/`top_k_idx`: `(n_tokens,
        num_experts_per_tok)`, the router's output in
        whatever order/normalization that architecture's router produces (this class only reads the
        values).
        """
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
    """Packs each expert into its own `QuantizedLinear` when `enabled` and a GEMV kernel exists for
    the tensor type, freeing
    the big dense placeholder once all are packed (module docstring); otherwise the eager
    `.copy_`. Returns whether it
    packed (the caller keeps the loader's mmap open past materialization only if at least one tensor
    did; `_mark_quantized_native_used`).
    """
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
            return False
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

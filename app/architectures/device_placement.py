from typing import ClassVar

import torch
from torch import nn

from app.server.errors import DeviceUnavailableError


class DevicePlacement:
    """Mixin for `ModelArchitecture`: which device the model computes on (CPU unless `place_on`).

    `SUPPORTS_GPU` is opt-in per architecture: only plain-attention forms whose per-forward
    tensors (masks, rope, caches) follow `input_ids`/the cache device are marked. Hybrid recurrent
    layers (Mamba-2, Gated DeltaNet) and MoE routing still build CPU tensors and stay CPU-only.
    """

    SUPPORTS_GPU: ClassVar[bool] = False
    _compute_device: torch.device = torch.device("cpu")
    _device_packed: bool = False

    @property
    def device(self) -> torch.device:
        return self._compute_device

    def place_on(self, device: torch.device, packed_weights: bool = False) -> None:
        """Moves a freshly built (still unmaterialized) model to `device`.

        `to_empty` reallocates every parameter/buffer there without copying the CPU placeholders;
        derived buffers (rope `inv_freq`) are recomputed, since `to_empty` leaves them garbage.
        Weights arrive later: `_materialize_weights`'s `.copy_()` reads the GGUF on the CPU and
        writes into the device tensors.

        `packed_weights` keeps supported quantized projections/embeddings packed on the device
        (`DevicePackedLinear`) instead of bf16. The big Linear/Embedding placeholders are then left
        where they are - uninitialized CPU storage, never touched so not resident - rather than
        allocated on the device up front, which would reserve the full bf16 model. Those the
        architecture doesn't replace get loaded in place and moved over by `settle_on_device`.
        It also works on the CPU device, which is how it is tested without a GPU.
        """
        if device.type == "cpu" and not packed_weights:
            return
        if not self.SUPPORTS_GPU:
            raise DeviceUnavailableError(
                f"architecture {type(self).__name__} has no GPU support yet (CPU only)"
            )
        big = {id(m.weight) for m in self.modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        for module in self.modules():  # e.g. MoE expert stacks, which hold their own big params
            big.update(id(p) for p in getattr(module, "deferred_params", list)())
        self._apply(
            lambda t: t if packed_weights and id(t) in big else torch.empty_like(t, device=device)
        )
        self._rebuild_derived_buffers()
        self._apply(lambda t: t.to(device))  # includes the rebuilt rope buffers
        if packed_weights:
            for module in self.modules():
                getattr(module, "use_device_packing", lambda _device: None)(device)
        self._compute_device = device
        self._device_packed = packed_weights

    def settle_on_device(self) -> None:
        """After materialization in packed mode: weights the architecture loaded the plain way
        (norms, anything that isn't a packable quant type) sit on the CPU placeholders - move them
        to the device. A no-op for tensors already there."""
        self._apply(lambda t: t.to(self._compute_device))

    def to_compute_device(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor.to(self._compute_device)

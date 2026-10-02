import torch

from app.server.errors import DeviceUnavailableError


class ComputeDevice:
    """Where a model's weights, caches and forward pass live: the CPU (default) or a CUDA GPU.

    GPU mode is EXPERIMENTAL - written without a GPU on the dev machine, so it has only been
    exercised through the CPU path and meta-device shape checks, never on real hardware.
    """

    def __init__(self, torch_device: torch.device) -> None:
        self._device = torch_device

    @classmethod
    def resolve(cls, spec: str) -> "ComputeDevice":
        """`"cpu"`, `"cuda"` or `"cuda:N"`. A GPU request that can't be met raises - it never
        silently falls back to the CPU, which is what `MATRICXON_DEVICE=cuda` used to do."""
        try:
            device = torch.device(spec)
        except (RuntimeError, ValueError) as error:
            raise DeviceUnavailableError(f"unknown device {spec!r}: {error}") from error
        if device.type == "cpu":
            return cls(device)
        if device.type != "cuda":
            raise DeviceUnavailableError(f"unsupported device {spec!r}: use 'cpu' or 'cuda[:N]'")
        if not torch.cuda.is_available():
            raise DeviceUnavailableError(
                f"device {spec!r} requested but PyTorch sees no CUDA GPU "
                "(CPU-only build, missing driver, or no GPU)"
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise DeviceUnavailableError(
                f"device {spec!r} requested but only {torch.cuda.device_count()} CUDA device(s)"
            )
        return cls(torch.device("cuda", index))

    @property
    def torch_device(self) -> torch.device:
        return self._device

    @property
    def is_gpu(self) -> bool:
        return self._device.type != "cpu"

    @property
    def name(self) -> str:
        return torch.cuda.get_device_name(self._device) if self.is_gpu else "cpu"

    def load_dtype(self) -> torch.dtype:
        """Weights dtype on a GPU: bf16 when the card supports it, else fp16."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def free_memory_bytes(self) -> int | None:
        """Free VRAM (not system RAM); None on the CPU, where `available_memory_bytes` applies."""
        if not self.is_gpu:
            return None
        free, _total = torch.cuda.mem_get_info(self._device)
        return free

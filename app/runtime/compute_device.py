import torch

from app.server.errors import DeviceUnavailableError


class ComputeDevice:
    """Where a model's weights, caches and forward pass live: the CPU (default), a CUDA GPU or an
    Apple Silicon GPU (MPS).

    GPU mode is EXPERIMENTAL: written without a GPU on the dev machine, exercised only through the
    CPU path and meta-device shape checks, never on real hardware.
    """

    def __init__(self, torch_device: torch.device) -> None:
        self._device = torch_device

    @classmethod
    def resolve(cls, spec: str) -> "ComputeDevice":
        """`"cpu"`, `"cuda"`, `"cuda:N"` or `"mps"`. A GPU request that can't be met raises; it
        never falls back silently to the CPU, as `MATRICXON_DEVICE=cuda` used to."""
        try:
            device = torch.device(spec)
        except (RuntimeError, ValueError) as error:
            raise DeviceUnavailableError(f"unknown device {spec!r}: {error}") from error
        if device.type == "cpu":
            return cls(device)
        if device.type == "mps":
            return cls._resolve_mps(spec)
        if device.type != "cuda":
            raise DeviceUnavailableError(
                f"unsupported device {spec!r}: use 'cpu', 'cuda[:N]' or 'mps'"
            )
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

    @classmethod
    def _resolve_mps(cls, spec: str) -> "ComputeDevice":
        if not torch.backends.mps.is_available():
            raise DeviceUnavailableError(
                f"device {spec!r} requested but PyTorch sees no Apple GPU "
                "(not Apple Silicon, macOS too old, or a build without MPS)"
            )
        return cls(torch.device("mps"))

    @property
    def torch_device(self) -> torch.device:
        return self._device

    @property
    def is_gpu(self) -> bool:
        return self._device.type != "cpu"

    @property
    def is_mps(self) -> bool:
        return self._device.type == "mps"

    @property
    def name(self) -> str:
        if not self.is_gpu:
            return "cpu"
        if self.is_mps:
            return "Apple GPU (MPS)"
        return torch.cuda.get_device_name(self._device)

    def load_dtype(self) -> torch.dtype:
        """Weights dtype on a GPU: bf16 when supported, else fp16."""
        if self.is_mps:
            return torch.bfloat16 if self._mps_supports_bf16() else torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @staticmethod
    def _mps_supports_bf16() -> bool:
        """MPS has bf16 from macOS 14; older versions fail on allocation."""
        try:
            torch.ones(1, dtype=torch.bfloat16, device="mps")
        except (RuntimeError, TypeError):
            return False
        return True

    def free_memory_bytes(self) -> int | None:
        """Free GPU memory; None on the CPU, where `available_memory_bytes` applies. On MPS (shared
        with system RAM): Metal's recommended working set minus what is already allocated."""
        if not self.is_gpu:
            return None
        if self.is_mps:
            return max(0, torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory())
        free, _total = torch.cuda.mem_get_info(self._device)
        return free

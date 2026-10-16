"""Apple Silicon (MPS) in ComputeDevice, without a Mac: torch's MPS calls are patched."""

import os

import pytest
import torch

from app.runtime.compute_device import ComputeDevice
from app.server.errors import DeviceUnavailableError

MPS = torch.device("mps")


def test_mps_without_an_apple_gpu_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="no Apple GPU"):
        ComputeDevice.resolve("mps")


def test_mps_resolves_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = ComputeDevice.resolve("mps")
    assert device.is_gpu and device.is_mps
    assert device.torch_device.type == "mps"
    assert device.name == "Apple GPU (MPS)"


def test_unknown_device_type_lists_mps() -> None:
    with pytest.raises(DeviceUnavailableError, match="'mps'"):
        ComputeDevice.resolve("xpu")


def test_mps_free_memory_is_working_set_minus_allocated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10_000)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 4_000)
    assert ComputeDevice(MPS).free_memory_bytes() == 6_000


def test_mps_free_memory_never_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1_000)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 5_000)
    assert ComputeDevice(MPS).free_memory_bytes() == 0


@pytest.mark.parametrize(("bf16", "expected"), [(True, torch.bfloat16), (False, torch.float16)])
def test_mps_load_dtype_follows_bf16_support(
    monkeypatch: pytest.MonkeyPatch, bf16: bool, expected: torch.dtype
) -> None:
    monkeypatch.setattr(ComputeDevice, "_mps_supports_bf16", staticmethod(lambda: bf16))
    assert ComputeDevice(MPS).load_dtype() == expected


def test_mps_cpu_fallback_is_on_by_default() -> None:
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

import torch
from torch import nn

from app.vision.device import VisionDevice


class _Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen: torch.device | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x.device
        return x * self.weight


def test_cpu_placement_is_a_no_op_and_encode_returns_cpu_tensors() -> None:
    probe = _Probe()
    assert VisionDevice.place(probe, torch.device("cpu")) is probe
    out = VisionDevice.encode(probe, torch.ones(2))
    assert out.device.type == "cpu" and probe.seen == torch.device("cpu")


def test_non_cpu_placement_moves_the_encoder() -> None:
    placed = VisionDevice.place(_Probe(), torch.device("meta"))
    assert next(placed.parameters()).device.type == "meta"

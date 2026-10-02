import torch
from torch import nn


class VisionDevice:
    """Runs a vision encoder on the same device as its language model (experimental GPU mode).

    Encoders are built on the CPU in float32 (unchanged); `place` moves one to `device` once, and
    `encode` feeds it pixels there and brings the embeddings back to the CPU, where the fusion code
    and `forward(image_embeddings=...)` expect them.
    """

    @staticmethod
    def place(encoder: nn.Module, device: torch.device) -> nn.Module:
        return encoder if device.type == "cpu" else encoder.to(device)

    @staticmethod
    def encode(encoder: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        device = next(encoder.parameters()).device
        return encoder(pixel_values.to(device)).cpu()

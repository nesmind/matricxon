import base64
import io
import math

import numpy as np
import torch
from PIL import Image


class Qwen3VLImagePreprocessor:
    """Base64 image -> normalized `(1, 3, H, W)` with H and W snapped to multiples of `factor`
    (`patch_size * merge`, 32), keeping aspect ratio - HF's `smart_resize`: round each side to the
    nearest multiple, then scale down/up if the area leaves `[min_pixels, max_pixels]`.
    `max_pixels` is lower than HF's default (16.7M) to bound ViT cost on this project's hardware:
    1M pixels is at most 1024 merged tokens."""

    def __init__(
        self,
        mean: list[float],
        std: list[float],
        factor: int = 32,
        min_pixels: int = 65536,
        max_pixels: int = 1024 * 32 * 32,
    ) -> None:
        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self._factor = factor
        self._min_pixels = min_pixels
        self._max_pixels = max_pixels

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        f = self._factor
        h, w = max(f, round(height / f) * f), max(f, round(width / f) * f)
        if h * w > self._max_pixels:
            scale = math.sqrt(height * width / self._max_pixels)
            h, w = (
                max(f, math.floor(height / scale / f) * f),
                max(f, math.floor(width / scale / f) * f),
            )
        elif h * w < self._min_pixels:
            scale = math.sqrt(self._min_pixels / (height * width))
            h, w = math.ceil(height * scale / f) * f, math.ceil(width * scale / f) * f
        return h, w

    def preprocess(self, image_base64: str) -> torch.Tensor:
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        h, w = self.target_size(image.height, image.width)
        image = image.resize((w, h), Image.BICUBIC)
        pixels = torch.from_numpy(np.array(image)).to(torch.float32) / 255.0
        return ((pixels.permute(2, 0, 1) - self._mean) / self._std).unsqueeze(0)

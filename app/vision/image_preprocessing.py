import base64
import io

import numpy as np
import torch
from PIL import Image


class ClipImagePreprocessor:
    """Decodes a base64 image (Ollama's real `ChatMessage.images` wire

    format) into the normalized `(1, 3, image_size, image_size)` tensor a
    `ClipVisionEncoder` expects - real preprocessing, not the "count them
    and ignore the bytes" placeholder every other architecture's
    `[IMG]`-token handling still uses today.

    `mean`/`std` come straight from the real GGUF's own
    `clip.vision.image_mean`/`image_std` metadata (confirmed `[0.5, 0.5,
    0.5]` on the real moondream2 mmproj file - not assumed to be
    ImageNet's more common `[0.485, 0.456, 0.406]`/`[0.229, 0.224, 0.225]`).
    A plain resize to `image_size x image_size` (no aspect-ratio-preserving
    crop) matches llama.cpp's own `clip.cpp` default preprocessing for a
    model with no separate crop/pad metadata key, which this GGUF doesn't
    carry.
    """

    def __init__(self, image_size: int, mean: list[float], std: list[float]) -> None:
        self._image_size = image_size
        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def preprocess(self, image_base64: str) -> torch.Tensor:
        raw_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image = image.resize((self._image_size, self._image_size), Image.BICUBIC)

        pixels = torch.from_numpy(np.array(image)).to(torch.float32) / 255.0  # (H, W, 3)
        pixels = pixels.permute(2, 0, 1)  # (3, H, W)
        normalized = (pixels - self._mean) / self._std
        return normalized.unsqueeze(0)  # (1, 3, H, W)

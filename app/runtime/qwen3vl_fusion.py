"""Qwen3.5 / Qwen3-VL image fusion: the counterpart of `vision_fusion.build_prompt_with_images`
for a `qwen3vl_merger` mmproj. Each `[IMG]` marker (see `ChatTemplatePromptBuilder`) becomes
`<|vision_start|>` + one `<|image_pad|>` per merged image token + `<|vision_end|>`, the image goes
through `Qwen3VLVisionEncoder`, and the prompt gets the model's 3-axis M-RoPE positions: text
tokens advance all three axes together; an image's tokens share one temporal index and take
row/column indices on the other two, after which text resumes at `start + max(rows, cols)` (HF
`get_rope_index`; llama.cpp's mtmd does the same). `position_delta` is how far later decode
positions sit below the cache length.
"""

from dataclasses import dataclass

import torch

from app.gguf.loader import GGUFModelLoader
from app.models.installed_model import InstalledModel
from app.runtime.vision_fusion import IMAGE_MARKER
from app.vision.qwen3vl_encoder import Qwen3VLVisionEncoder
from app.vision.qwen3vl_preprocessing import Qwen3VLImagePreprocessor

_VISION_START, _VISION_END, _IMAGE_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
_encoder_cache: dict[str, Qwen3VLVisionEncoder] = {}


@dataclass(frozen=True)
class FusedPrompt:
    token_ids: list[int]
    image_embeddings: list[tuple[int, torch.Tensor]]
    position_ids: torch.Tensor  # (3, len(token_ids))
    position_delta: int


def _get_encoder(mmproj: InstalledModel) -> Qwen3VLVisionEncoder:
    if mmproj.tag not in _encoder_cache:
        loader = GGUFModelLoader(mmproj.path, dtype=torch.float32)
        _encoder_cache[mmproj.tag] = Qwen3VLVisionEncoder.from_gguf(loader)
    return _encoder_cache[mmproj.tag]


def build_qwen3vl_prompt(
    prompt: str, images_b64: list[str], tokenizer: object, mmproj: InstalledModel
) -> FusedPrompt:
    """One `[IMG]` marker per image, in order (a marker without an image is dropped)."""
    encoder = _get_encoder(mmproj)
    preprocessor = Qwen3VLImagePreprocessor(encoder.image_mean, encoder.image_std)
    pad_id = tokenizer.encode(_IMAGE_PAD, add_bos=False)[0]
    segments = prompt.split(IMAGE_MARKER)

    token_ids: list[int] = []
    positions: list[torch.Tensor] = []  # each (3, n)
    embeddings: list[tuple[int, torch.Tensor]] = []
    next_pos = 0

    def add_text(text: str) -> None:
        nonlocal next_pos
        ids = tokenizer.encode(text, add_bos=False)
        token_ids.extend(ids)
        positions.append((next_pos + torch.arange(len(ids))).expand(3, -1))
        next_pos += len(ids)

    with torch.no_grad():
        for i, segment in enumerate(segments):
            has_image = i < len(segments) - 1 and i < len(images_b64)
            add_text(
                ("" if i == 0 else _VISION_END) + segment + (_VISION_START if has_image else "")
            )
            if not has_image:
                continue
            pixels = preprocessor.preprocess(images_b64[i])
            rows, cols = encoder.merged_grid(pixels.shape[-2], pixels.shape[-1])
            embeddings.append((len(token_ids), encoder(pixels)))
            token_ids.extend([pad_id] * (rows * cols))
            axes = torch.stack(
                [
                    torch.full((rows * cols,), next_pos),
                    next_pos + torch.arange(rows).repeat_interleave(cols),
                    next_pos + torch.arange(cols).repeat(rows),
                ]
            )
            positions.append(axes)
            next_pos += max(rows, cols)

    position_ids = torch.cat(positions, dim=1)
    return FusedPrompt(token_ids, embeddings, position_ids, next_pos - len(token_ids))

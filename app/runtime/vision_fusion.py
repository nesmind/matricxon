"""LLaVA vision support (`LlamaArchitecture` only; `ModelArchitecture.forward`): decodes a chat
request's base64 images, runs each through the paired mmproj's `ClipVisionEncoder` and expands the
prompt's literal `[IMG]` markers (`Mistral3PromptBuilder`'s output) into per-patch placeholder token
positions, so `LlamaArchitecture._forward_impl` knows which spans to overwrite with image
embeddings.

Scoped to one base-resolution view per image: LLaVA-1.6's "AnyRes" multi-crop scheme (variable crops
per aspect ratio, spatial reassembly, a learned row-separator token; the mmproj has
`clip.vision.image_aspect_ratio = anyres`) is deliberately deferred.
"""

import torch

from app.gguf.loader import GGUFModelLoader
from app.models.installed_model import InstalledModel
from app.vision.clip_vision_encoder import ClipVisionEncoder
from app.vision.device import VisionDevice
from app.vision.image_preprocessing import ClipImagePreprocessor

IMAGE_MARKER = "[IMG]"

# Weights (624MB for the LLaVA mmproj) loaded once per mmproj tag and kept for the process lifetime:
# reloading+eval each chat turn would waste time. A plain module-level cache, not part of
# ModelManager's load/evict lifecycle, as ClipVisionEncoder isn't a ModelArchitecture (its
# docstring).
_encoder_cache: dict[tuple[str, str], ClipVisionEncoder] = {}


def _get_encoder(mmproj: InstalledModel, device: torch.device) -> ClipVisionEncoder:
    key = (mmproj.tag, str(device))
    if key not in _encoder_cache:
        loader = GGUFModelLoader(mmproj.path, dtype=torch.float32)
        _encoder_cache[key] = VisionDevice.place(ClipVisionEncoder.from_gguf(loader), device)
    return _encoder_cache[key]


def build_prompt_with_images(
    prompt: str,
    images_b64: list[str],
    tokenizer: object,
    mmproj: InstalledModel,
    device: torch.device | None = None,
) -> tuple[list[int], list[tuple[int, torch.Tensor]]]:
    """Splits `prompt` on the literal `[IMG]` marker (one per image, in message order), tokenizes
    each text segment independently (so BOS lands once at the start, as a plain
    `tokenizer.encode(prompt, add_bos=True)` would) and inserts `encoder.num_patches` placeholder
    token ids at each marker. The placeholder values never matter (`LlamaArchitecture._forward_impl`
    overwrites them with image embeddings before any lookup); `0` is an always-valid in-vocab id.

    Assumes one `[IMG]` marker per image, in order: true for every caller (chat_router.py uses this
    only for a request with both images and a paired mmproj; everything else uses plain
    `tokenizer.encode`).
    """
    encoder = _get_encoder(mmproj, device or torch.device("cpu"))
    preprocessor = ClipImagePreprocessor(encoder.image_size, encoder.image_mean, encoder.image_std)

    segments = prompt.split(IMAGE_MARKER)
    token_ids: list[int] = []
    image_embeddings: list[tuple[int, torch.Tensor]] = []
    with torch.no_grad():
        for i, segment in enumerate(segments):
            token_ids.extend(tokenizer.encode(segment, add_bos=(i == 0)))
            if i < len(images_b64):
                pixel_values = preprocessor.preprocess(images_b64[i])
                patch_embeds = VisionDevice.encode(encoder, pixel_values)[0]
                start = len(token_ids)
                token_ids.extend([0] * encoder.num_patches)
                image_embeddings.append((start, patch_embeds))
    return token_ids, image_embeddings

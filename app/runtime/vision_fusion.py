"""Real LLaVA vision support (2026-09-21, `LlamaArchitecture` only - see
`ModelArchitecture.forward`'s own docstring): decodes a real chat request's base64 images, runs
each through the paired mmproj's `ClipVisionEncoder`, and expands the prompt's literal `[IMG]`
markers (`Mistral3PromptBuilder`'s own existing output - see that class) into real per-patch
placeholder token positions, so `LlamaArchitecture._forward_impl` knows exactly which spans to
overwrite with real image embeddings.

Scoped to a single base-resolution view per image - LLaVA-1.6's real "AnyRes" multi-crop scheme
(variable crop count per image aspect ratio, spatial reassembly, a learned row-separator token,
confirmed via the real mmproj's own `clip.vision.image_aspect_ratio = anyres` metadata) is real,
substantially complex, deliberately deferred follow-up work - not implemented here.
"""

import torch

from app.gguf.loader import GGUFModelLoader
from app.models.installed_model import InstalledModel
from app.vision.clip_vision_encoder import ClipVisionEncoder
from app.vision.image_preprocessing import ClipImagePreprocessor

IMAGE_MARKER = "[IMG]"

# Real weights (624MB for the real LLaVA mmproj) loaded once per real mmproj tag and kept for this
# process's lifetime - re-loading+re-running eval() on every chat turn would be wasteful, and this
# stays a plain module-level cache rather than joining ModelManager's own load/evict lifecycle,
# same as ClipVisionEncoder itself deliberately isn't a ModelArchitecture (see its own docstring).
_encoder_cache: dict[str, ClipVisionEncoder] = {}


def _get_encoder(mmproj: InstalledModel) -> ClipVisionEncoder:
    if mmproj.tag not in _encoder_cache:
        loader = GGUFModelLoader(mmproj.path, dtype=torch.float32)
        _encoder_cache[mmproj.tag] = ClipVisionEncoder.from_gguf(loader)
    return _encoder_cache[mmproj.tag]


def build_prompt_with_images(
    prompt: str, images_b64: list[str], tokenizer: object, mmproj: InstalledModel
) -> tuple[list[int], list[tuple[int, torch.Tensor]]]:
    """Splits `prompt` on the real literal `[IMG]` marker text `Mistral3PromptBuilder` already
    emits (one per real image, in message order), tokenizes each real text segment independently
    (so BOS still lands exactly once, at the very start - matching a plain `tokenizer.encode(
    prompt, add_bos=True)` call's own real semantics), and inserts `encoder.num_patches`
    placeholder token ids at each marker position. The placeholder ids' own value never matters -
    see `LlamaArchitecture._forward_impl`'s own docstring: they're overwritten with real image
    embeddings before ever being looked up for real - `0` is used here purely because it's always
    a real, valid in-vocab id.

    Assumes exactly one `[IMG]` marker per real image, in the same order - true for every existing
    caller today (see `chat_router.py`: only a request with both real images *and* a paired mmproj
    routes through this function at all; everything else still calls plain `tokenizer.encode()`).
    """
    encoder = _get_encoder(mmproj)
    preprocessor = ClipImagePreprocessor(encoder.image_size, encoder.image_mean, encoder.image_std)

    segments = prompt.split(IMAGE_MARKER)
    token_ids: list[int] = []
    image_embeddings: list[tuple[int, torch.Tensor]] = []
    with torch.no_grad():
        for i, segment in enumerate(segments):
            token_ids.extend(tokenizer.encode(segment, add_bos=(i == 0)))
            if i < len(images_b64):
                pixel_values = preprocessor.preprocess(images_b64[i])
                patch_embeds = encoder(pixel_values)[0]  # drop the batch dim: (num_patches, n_embd)
                start = len(token_ids)
                token_ids.extend([0] * encoder.num_patches)
                image_embeddings.append((start, patch_embeds))
    return token_ids, image_embeddings

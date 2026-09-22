"""Manual smoke test for M10: ClipVisionEncoder against the real moondream2

mmproj GGUF (910MB, real weights) - real image preprocessing (base64 PNG ->
normalized pixel tensor) through the real 28-layer SigLIP-shaped ViT +
MLP projector, checked for shape correctness, no NaN/Inf, and that two
different real images actually produce different embeddings (not a
degenerate/collapsed output).

Not a pass/fail check for the same reason as the other manual_*_check.py scripts - checks shape/
finiteness/non-degeneracy only, not output correctness against a real text generation (see
tests/integration/test_real_fixture_moondream2.py for the full real end-to-end phi2 + vision
generation check now that `phi2` is a real, supported architecture).

    .venv/bin/python -m scripts.manual_vision_check
"""

import argparse
import base64
import io

import torch
from PIL import Image, ImageDraw

from app.gguf.loader import GGUFModelLoader
from app.vision.clip_vision_encoder import ClipVisionEncoder
from app.vision.image_preprocessing import ClipImagePreprocessor

DEFAULT_MMPROJ_PATH = "data/models/hf.co/moondream/moondream2-gguf/mmproj.gguf"


def _real_test_image_base64(fill: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (200, 200), fill)
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 50, 150, 150), fill=(255 - fill[0], 255 - fill[1], 255 - fill[2]))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmproj-path", default=DEFAULT_MMPROJ_PATH)
    args = parser.parse_args()

    loader = GGUFModelLoader(args.mmproj_path, dtype=torch.float32)
    model = ClipVisionEncoder.from_gguf(loader)
    preprocessor = ClipImagePreprocessor(model.image_size, model.image_mean, model.image_std)
    print(
        f"loaded: {model.n_layer} layers, n_embd={model.n_embd}, "
        f"num_patches={model.num_patches}, projection_dim={model.projection_dim}"
    )

    red_image = _real_test_image_base64((255, 0, 0))
    green_image = _real_test_image_base64((0, 255, 0))

    with torch.no_grad():
        red_out = model(preprocessor.preprocess(red_image))
        green_out = model(preprocessor.preprocess(green_image))
        red_out_again = model(preprocessor.preprocess(red_image))

    print(f"output shape: {tuple(red_out.shape)}")
    has_nan, has_inf = torch.isnan(red_out).any().item(), torch.isinf(red_out).any().item()
    print(f"has nan: {has_nan}, has inf: {has_inf}")
    print(f"mean: {red_out.mean().item():.4f}, std: {red_out.std().item():.4f}")
    print(
        f"different real images produce different output: {not torch.allclose(red_out, green_out)}"
    )
    print(f"same real image is deterministic: {torch.equal(red_out, red_out_again)}")


if __name__ == "__main__":
    main()

"""Shared chest-X-ray loading that preserves high-bit-depth PNG contrast."""

from pathlib import Path

import numpy as np
from PIL import Image


HIGH_BIT_MODES = {"I", "I;16", "I;16L", "I;16B"}


def normalize_cxr_image(image: Image.Image) -> Image.Image:
    """Convert a PIL image to RGB without saturating 16-bit grayscale pixels.

    PadChest-GR PNGs are decoded by Pillow as 32-bit integer arrays whose values
    represent 16-bit grayscale. PIL's direct I -> RGB conversion clips values
    above 255 and produces an almost-white image. Dividing by 256 preserves the
    original 16-bit intensity mapping as 8-bit grayscale.
    """
    if image.mode in HIGH_BIT_MODES:
        array = np.asarray(image)
        if array.size and float(array.max()) > 255:
            array = (array.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="L").convert("RGB")
    return image.convert("RGB")


def load_cxr_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return normalize_cxr_image(image)


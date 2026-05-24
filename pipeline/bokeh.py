from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from pipeline.depth import normalize_depth
from pipeline.image_io import ensure_rgb
from pipeline.mask import _binary_mask_array


def _foreground_mask_image(mask: np.ndarray | Image.Image, size: tuple[int, int]) -> Image.Image:
    if isinstance(mask, Image.Image):
        mask_image = mask.convert("L")
    else:
        mask_image = Image.fromarray(_binary_mask_array(mask), mode="L")

    if mask_image.size != size:
        mask_image = mask_image.resize(size, Image.Resampling.NEAREST)

    return mask_image.point(lambda pixel: 255 if pixel >= 128 else 0, mode="L")


def apply_depth_bokeh(
    image: Image.Image,
    depth: np.ndarray,
    foreground_mask: np.ndarray | Image.Image,
    blur_radius: int = 12,
) -> Image.Image:
    rgb = ensure_rgb(image)
    normalized = normalize_depth(depth)
    if normalized.shape != (rgb.height, rgb.width):
        normalized = cv2.resize(normalized, rgb.size, interpolation=cv2.INTER_LINEAR)

    far_blur = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mid_blur = rgb.filter(ImageFilter.GaussianBlur(radius=max(2, blur_radius // 2)))

    depth_pixels = (normalized * 255.0).round().astype(np.uint8)
    far_mask = Image.fromarray((depth_pixels < 90).astype(np.uint8) * 255, mode="L")
    mid_mask = Image.fromarray(
        ((depth_pixels >= 90) & (depth_pixels < 170)).astype(np.uint8) * 255,
        mode="L",
    )

    composed = rgb.copy()
    composed.paste(mid_blur, mask=mid_mask)
    composed.paste(far_blur, mask=far_mask)

    fg_mask = _foreground_mask_image(foreground_mask, rgb.size)
    composed.paste(rgb, mask=fg_mask)
    return composed

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from pipeline.depth import normalize_depth
from pipeline.image_io import ensure_rgb


def _binary_mask_array(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.dtype == np.bool_:
        return values.astype(np.uint8) * 255

    if np.issubdtype(values.dtype, np.floating):
        finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        if float(finite.max(initial=0.0)) <= 1.0:
            return (finite >= 0.5).astype(np.uint8) * 255
        return (finite >= 127.5).astype(np.uint8) * 255

    return (values.astype(np.uint8) >= 128).astype(np.uint8) * 255


def _morphology_kernel(shape: tuple[int, int]) -> np.ndarray:
    shortest_side = min(shape)
    if shortest_side < 5:
        return np.ones((1, 1), dtype=np.uint8)
    size = min(5, shortest_side)
    if size % 2 == 0:
        size -= 1
    return np.ones((size, size), dtype=np.uint8)


def compute_scharr_gradient(depth: np.ndarray) -> np.ndarray:
    normalized = normalize_depth(depth)
    gradient_x = cv2.Scharr(normalized, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(normalized, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    return normalize_depth(magnitude)


def build_foreground_mask(depth: np.ndarray) -> np.ndarray:
    normalized = normalize_depth(depth)

    base_threshold = np.percentile(normalized, 60)
    base_mask = normalized > base_threshold

    gradient = compute_scharr_gradient(normalized)
    edge_threshold = np.percentile(gradient, 85)
    edge_mask = gradient > edge_threshold

    combined = np.where(base_mask | edge_mask, 255, 0).astype(np.uint8)
    kernel = _morphology_kernel(combined.shape)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    if min(combined.shape) < 12:
        return np.where(closed > 0, 255, 0).astype(np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return np.where(opened > 0, 255, 0).astype(np.uint8)


def extract_subject(image: Image.Image, mask: np.ndarray | Image.Image) -> Image.Image:
    rgb_image = ensure_rgb(image)

    if isinstance(mask, Image.Image):
        mask_image = mask.convert("L")
    else:
        mask_image = Image.fromarray(_binary_mask_array(mask), mode="L")

    if mask_image.size != rgb_image.size:
        mask_image = mask_image.resize(rgb_image.size, Image.Resampling.NEAREST)

    background = Image.new("RGB", rgb_image.size, color=(255, 255, 255))
    background.paste(rgb_image, mask=mask_image)
    return background

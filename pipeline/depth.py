from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

from pipeline.image_io import ensure_rgb

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    values = depth.astype(np.float32)
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    span = maximum - minimum
    if span <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - minimum) / span
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def depth_array_to_image(depth: np.ndarray) -> Image.Image:
    normalized = normalize_depth(depth)
    pixels = (normalized * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


@lru_cache(maxsize=1)
def load_depth_pipeline() -> Any:
    from transformers import pipeline

    return pipeline("depth-estimation", model=DEPTH_MODEL_ID)


def estimate_depth(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    depth_pipe = load_depth_pipeline()
    output = depth_pipe(ensure_rgb(image))
    predicted_depth = output["predicted_depth"]

    if hasattr(predicted_depth, "detach"):
        depth_array = predicted_depth.detach().cpu().numpy()
    else:
        depth_array = np.asarray(predicted_depth)

    depth_array = np.squeeze(depth_array).astype(np.float32)
    normalized = normalize_depth(depth_array)
    return normalized, depth_array_to_image(normalized)

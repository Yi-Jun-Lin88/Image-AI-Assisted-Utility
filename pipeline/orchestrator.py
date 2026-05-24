from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from pipeline.bokeh import apply_depth_bokeh
from pipeline.caption import fallback_caption, generate_caption
from pipeline.classify import classify_image
from pipeline.depth import estimate_depth
from pipeline.image_io import ensure_rgb, resize_for_inference
from pipeline.mask import build_foreground_mask, extract_subject
from pipeline.types import CaptionResult, ClassificationResult, PipelineResult

DepthFn = Callable[[Image.Image], tuple[np.ndarray, Image.Image]]
ClassifyFn = Callable[[Image.Image], list[ClassificationResult]]


def run_image_pipeline(
    image: Image.Image,
    *,
    api_key: str | None = None,
    caption_provider: str = "fallback",
    depth_fn: DepthFn = estimate_depth,
    classify_fn: ClassifyFn = classify_image,
) -> PipelineResult:
    original = resize_for_inference(ensure_rgb(image))
    errors: list[str] = []

    depth_array: np.ndarray | None = None
    depth_map: Image.Image | None = None
    subject: Image.Image | None = None
    bokeh: Image.Image | None = None

    try:
        depth_array, depth_map = depth_fn(original)
    except Exception as exc:
        errors.append(f"Depth estimation failed: {exc}")

    if depth_array is not None:
        try:
            foreground_mask = build_foreground_mask(depth_array)
        except Exception as exc:
            errors.append(f"Depth-aware image processing failed: {exc}")
            foreground_mask = None

        if foreground_mask is not None:
            try:
                subject = extract_subject(original, foreground_mask)
            except Exception as exc:
                errors.append(f"Subject extraction failed: {exc}")

            try:
                bokeh = apply_depth_bokeh(original, depth_array, foreground_mask)
            except Exception as exc:
                errors.append(f"Bokeh generation failed: {exc}")

    analysis_image = subject or original

    try:
        caption = generate_caption(analysis_image, api_key=api_key, provider=caption_provider)
    except Exception as exc:
        caption = fallback_caption()
        errors.append(f"Caption generation failed: {exc}")

    try:
        classifications = classify_fn(analysis_image)
    except Exception as exc:
        classifications = [ClassificationResult(label="image", score=1.0, used_fallback=True)]
        errors.append(f"Image classification failed: {exc}")
    else:
        if any(item.used_fallback for item in classifications):
            errors.append("Image classification used fallback output.")

    return PipelineResult(
        original=original,
        depth_map=depth_map,
        subject=subject,
        bokeh=bokeh,
        caption=caption,
        classifications=classifications,
        errors=errors,
    )

from __future__ import annotations

from PIL import Image

from pipeline.types import CaptionResult


def fallback_caption() -> CaptionResult:
    return CaptionResult(
        text=(
            "A foreground subject extracted from the uploaded image, prepared for "
            "background removal and portrait-style image editing."
        ),
        provider="fallback",
        used_fallback=True,
    )


def generate_caption(
    image: Image.Image, api_key: str | None = None, provider: str = "fallback"
) -> CaptionResult:
    return fallback_caption()

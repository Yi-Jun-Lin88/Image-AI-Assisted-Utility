from PIL import Image

from pipeline.image_io import (
    SUPPORTED_UPLOAD_TYPES,
    ensure_rgb,
    image_to_png_bytes,
    resize_for_inference,
)


def test_supported_upload_types_include_heif_formats() -> None:
    assert "heif" in SUPPORTED_UPLOAD_TYPES
    assert "heic" in SUPPORTED_UPLOAD_TYPES


def test_ensure_rgb_converts_alpha_image() -> None:
    image = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
    result = ensure_rgb(image)
    assert result.mode == "RGB"
    assert result.size == (10, 10)
    assert result.getpixel((0, 0)) == (255, 127, 127)


def test_ensure_rgb_flattens_luminance_alpha_image() -> None:
    image = Image.new("LA", (10, 10), color=(0, 128))
    result = ensure_rgb(image)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (127, 127, 127)


def test_resize_for_inference_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (2000, 1000), color=(10, 20, 30))
    result = resize_for_inference(image, max_side=500)
    assert result.size == (500, 250)


def test_resize_for_inference_does_not_upscale(sample_rgb_image: Image.Image) -> None:
    result = resize_for_inference(sample_rgb_image, max_side=500)
    assert result.size == sample_rgb_image.size


def test_image_to_png_bytes_exports_png(sample_rgb_image: Image.Image) -> None:
    data = image_to_png_bytes(sample_rgb_image)
    assert data.startswith(b"\x89PNG")
    assert len(data) > 50

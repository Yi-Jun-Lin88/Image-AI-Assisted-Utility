from __future__ import annotations

from io import BytesIO

from PIL import Image

SUPPORTED_UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "heif", "heic"]


def register_supported_image_openers() -> None:
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, color=(255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    background = Image.new("RGB", image.size, color=(255, 255, 255))
    return image.convert("RGB")


def resize_for_inference(image: Image.Image, max_side: int = 768) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

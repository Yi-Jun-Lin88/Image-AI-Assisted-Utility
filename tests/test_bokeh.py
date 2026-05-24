import numpy as np
from PIL import Image

from pipeline.bokeh import apply_depth_bokeh


def test_apply_depth_bokeh_preserves_size_and_mode() -> None:
    image = Image.new("RGB", (32, 24), color=(100, 120, 140))
    depth = np.zeros((24, 32), dtype=np.float32)
    mask = np.ones((24, 32), dtype=np.uint8) * 255

    result = apply_depth_bokeh(image, depth, mask)

    assert result.size == image.size
    assert result.mode == "RGB"


def test_apply_depth_bokeh_keeps_foreground_pixel_when_masked() -> None:
    image = Image.new("RGB", (10, 10), color=(20, 40, 60))
    depth = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255

    result = apply_depth_bokeh(image, depth, mask)

    assert result.getpixel((5, 5)) == (20, 40, 60)


def test_apply_depth_bokeh_changes_patterned_background() -> None:
    image = Image.new("RGB", (32, 32), color=(0, 0, 0))
    for x in range(32):
        for y in range(32):
            if (x + y) % 2 == 0:
                image.putpixel((x, y), (255, 255, 255))

    depth = np.zeros((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[14:18, 14:18] = 255

    result = apply_depth_bokeh(image, depth, mask, blur_radius=6)

    assert result.getpixel((0, 0)) != image.getpixel((0, 0))
    assert result.getpixel((15, 15)) == image.getpixel((15, 15))


def test_apply_depth_bokeh_accepts_wide_integer_mask() -> None:
    image = Image.new("RGB", (8, 8), color=(255, 255, 255))
    for x in range(8):
        for y in range(8):
            if (x + y) % 2 == 0:
                image.putpixel((x, y), (0, 0, 0))
    image.putpixel((4, 4), (255, 0, 0))

    depth = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint16)
    mask[4, 4] = 256

    result = apply_depth_bokeh(image, depth, mask)

    assert result.getpixel((4, 4)) == (255, 0, 0)

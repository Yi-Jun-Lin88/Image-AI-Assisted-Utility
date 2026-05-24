import numpy as np
from PIL import Image

from pipeline.mask import build_foreground_mask, compute_scharr_gradient, extract_subject


def test_compute_scharr_gradient_detects_vertical_edge() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[:, 10:] = 1.0

    gradient = compute_scharr_gradient(depth)

    assert gradient.shape == depth.shape
    assert gradient.dtype == np.float32
    assert gradient[:, 9:11].mean() > 0.8
    assert gradient[:, :5].max() == 0.0
    assert gradient[:, 15:].max() == 0.0


def test_build_foreground_mask_returns_binary_mask_with_center_foreground() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[6:14, 6:14] = 1.0

    mask = build_foreground_mask(depth)

    assert mask.dtype == np.uint8
    assert mask.shape == depth.shape
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_build_foreground_mask_preserves_small_foreground_region() -> None:
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[3:5, 3:5] = 1.0

    mask = build_foreground_mask(depth)

    assert mask[3:5, 3:5].max() == 255
    assert set(np.unique(mask)).issubset({0, 255})


def test_build_foreground_mask_subject_strength_controls_mask_size() -> None:
    depth = np.tile(np.linspace(0.0, 1.0, 30, dtype=np.float32), (30, 1))

    loose_mask = build_foreground_mask(depth, subject_strength=35)
    tight_mask = build_foreground_mask(depth, subject_strength=80)

    assert loose_mask.sum() > tight_mask.sum()


def test_extract_subject_uses_white_background_and_keeps_foreground_pixels() -> None:
    image = Image.new("RGB", (4, 4), color=(20, 40, 60))
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    result = extract_subject(image, mask)

    assert result.mode == "RGB"
    assert result.size == image.size
    assert result.getpixel((1, 1)) == (20, 40, 60)
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_extract_subject_accepts_bool_mask() -> None:
    image = Image.new("RGB", (4, 4), color=(20, 40, 60))
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    result = extract_subject(image, mask)

    assert result.getpixel((1, 1)) == (20, 40, 60)
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_extract_subject_accepts_zero_one_float_mask() -> None:
    image = Image.new("RGB", (4, 4), color=(20, 40, 60))
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1:3, 1:3] = 1.0

    result = extract_subject(image, mask)

    assert result.getpixel((1, 1)) == (20, 40, 60)
    assert result.getpixel((0, 0)) == (255, 255, 255)


def test_extract_subject_resizes_mask_without_blended_edges() -> None:
    image = Image.new("RGB", (8, 8), color=(20, 40, 60))
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255

    result = extract_subject(image, mask)

    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((3, 3)) == (20, 40, 60)

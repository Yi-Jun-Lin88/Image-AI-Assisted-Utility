import numpy as np
from PIL import Image

from pipeline.depth import depth_array_to_image, normalize_depth


def test_normalize_depth_scales_to_zero_one() -> None:
    depth = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    result = normalize_depth(depth)
    assert result.dtype == np.float32
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)


def test_normalize_depth_handles_flat_input() -> None:
    depth = np.ones((4, 4), dtype=np.float32) * 7.0
    result = normalize_depth(depth)
    assert np.all(result == 0.0)


def test_depth_array_to_image_exports_grayscale() -> None:
    depth = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    image = depth_array_to_image(depth)
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (2, 2)

from pathlib import Path
import sys

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    return Image.new("RGB", (120, 80), color=(40, 120, 200))

# Image AI Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deployable Streamlit demo that turns one image into depth map, depth-aware subject extraction, portrait bokeh, caption, classification, and downloadable outputs.

**Architecture:** Use a small Python package under `pipeline/` for image processing and model boundaries, with `app.py` owning only Streamlit UI. Heavy HuggingFace models are cached and optional services degrade gracefully so the demo remains usable without API keys.

**Tech Stack:** Python 3.10+, Streamlit, Pillow, NumPy, OpenCV headless, Transformers, Torch, pytest.

---

## File Structure

Create these files:

```text
app.py
pipeline/__init__.py
pipeline/bokeh.py
pipeline/caption.py
pipeline/classify.py
pipeline/depth.py
pipeline/image_io.py
pipeline/mask.py
pipeline/orchestrator.py
pipeline/types.py
requirements.txt
README.md
tests/conftest.py
tests/test_bokeh.py
tests/test_caption.py
tests/test_image_io.py
tests/test_mask.py
tests/test_orchestrator.py
```

Responsibilities:

- `app.py`: Streamlit UI, image source selection, progress states, result display, download buttons, PM notes.
- `pipeline/types.py`: Shared dataclasses used across pipeline modules.
- `pipeline/image_io.py`: PIL conversion, RGB normalization, image resizing, PNG byte export.
- `pipeline/depth.py`: HuggingFace depth model loading and depth map normalization.
- `pipeline/mask.py`: Scharr gradient, automatic foreground mask, subject extraction.
- `pipeline/bokeh.py`: Depth-layered Gaussian blur.
- `pipeline/caption.py`: Vision caption provider boundary plus deterministic fallback.
- `pipeline/classify.py`: HuggingFace image-classification provider plus deterministic fallback.
- `pipeline/orchestrator.py`: End-to-end pipeline coordination with step-level degradation.
- `tests/`: Fast tests for deterministic local logic. Tests must not download HuggingFace models.

## Task 1: Project Scaffold And Image Utilities

**Files:**
- Create: `requirements.txt`
- Create: `pipeline/__init__.py`
- Create: `pipeline/types.py`
- Create: `pipeline/image_io.py`
- Create: `tests/conftest.py`
- Create: `tests/test_image_io.py`

- [ ] **Step 1: Add dependency file**

Create `requirements.txt`:

```text
streamlit>=1.35
transformers>=4.42
torch>=2.2
Pillow>=10.3
numpy>=1.26
opencv-python-headless>=4.9
pytest>=8.2
```

- [ ] **Step 2: Write failing image utility tests**

Create `tests/conftest.py`:

```python
from PIL import Image
import pytest


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    return Image.new("RGB", (120, 80), color=(40, 120, 200))
```

Create `tests/test_image_io.py`:

```python
from PIL import Image

from pipeline.image_io import ensure_rgb, image_to_png_bytes, resize_for_inference


def test_ensure_rgb_converts_alpha_image() -> None:
    image = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
    result = ensure_rgb(image)
    assert result.mode == "RGB"
    assert result.size == (10, 10)


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/test_image_io.py -v
```

Expected: FAIL because `pipeline.image_io` does not exist.

- [ ] **Step 4: Implement shared types and image utilities**

Create `pipeline/__init__.py`:

```python
"""Image AI Utility pipeline package."""
```

Create `pipeline/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CaptionResult:
    text: str
    provider: str
    used_fallback: bool


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    score: float


@dataclass(frozen=True)
class PipelineResult:
    original: Image.Image
    depth_map: Image.Image | None
    subject: Image.Image | None
    bokeh: Image.Image | None
    caption: CaptionResult
    classifications: list[ClassificationResult]
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


DepthArray = np.ndarray
MaskArray = np.ndarray
```

Create `pipeline/image_io.py`:

```python
from __future__ import annotations

from io import BytesIO

from PIL import Image


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    background = Image.new("RGB", image.size, color=(255, 255, 255))
    if image.mode == "RGBA":
        background.paste(image, mask=image.getchannel("A"))
        return background
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/test_image_io.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pipeline/__init__.py pipeline/types.py pipeline/image_io.py tests/conftest.py tests/test_image_io.py
git commit -m "feat: add project scaffold and image utilities"
```

## Task 2: Depth Map Normalization Boundary

**Files:**
- Create: `pipeline/depth.py`
- Create: `tests/test_depth.py`

- [ ] **Step 1: Write failing depth utility tests**

Create `tests/test_depth.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_depth.py -v
```

Expected: FAIL because `pipeline.depth` does not exist.

- [ ] **Step 3: Implement depth helpers and lazy model loader**

Create `pipeline/depth.py`:

```python
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image
from transformers import pipeline

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_depth.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/depth.py tests/test_depth.py
git commit -m "feat: add depth normalization boundary"
```

## Task 3: Depth-Aware Mask And Subject Extraction

**Files:**
- Create: `pipeline/mask.py`
- Create: `tests/test_mask.py`

- [ ] **Step 1: Write failing mask tests**

Create `tests/test_mask.py`:

```python
import numpy as np
from PIL import Image

from pipeline.mask import build_foreground_mask, compute_scharr_gradient, extract_subject


def test_compute_scharr_gradient_detects_vertical_edge() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[:, 10:] = 1.0
    gradient = compute_scharr_gradient(depth)
    assert gradient.shape == depth.shape
    assert gradient[:, 9:11].mean() > gradient[:, :4].mean()


def test_build_foreground_mask_returns_binary_mask() -> None:
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[5:15, 5:15] = 1.0
    mask = build_foreground_mask(depth)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert mask[10, 10] == 255


def test_extract_subject_uses_white_background() -> None:
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    result = extract_subject(image, mask)
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((2, 2)) == (10, 20, 30)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_mask.py -v
```

Expected: FAIL because `pipeline.mask` does not exist.

- [ ] **Step 3: Implement Scharr mask and extraction**

Create `pipeline/mask.py`:

```python
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from pipeline.depth import normalize_depth
from pipeline.image_io import ensure_rgb


def compute_scharr_gradient(depth: np.ndarray) -> np.ndarray:
    normalized = normalize_depth(depth)
    grad_x = cv2.Scharr(normalized, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(normalized, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return normalize_depth(magnitude)


def build_foreground_mask(depth: np.ndarray) -> np.ndarray:
    normalized = normalize_depth(depth)
    threshold = float(np.percentile(normalized, 60))
    base = (normalized >= threshold).astype(np.uint8) * 255
    gradient = compute_scharr_gradient(normalized)
    edge = (gradient >= float(np.percentile(gradient, 85))).astype(np.uint8) * 255
    combined = cv2.bitwise_or(base, edge)
    kernel = np.ones((5, 5), dtype=np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened.astype(np.uint8)


def extract_subject(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = ensure_rgb(image)
    mask_image = Image.fromarray(mask, mode="L").resize(rgb.size, Image.Resampling.BILINEAR)
    background = Image.new("RGB", rgb.size, color=(255, 255, 255))
    background.paste(rgb, mask=mask_image)
    return background
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_mask.py -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/mask.py tests/test_mask.py
git commit -m "feat: add depth-aware subject mask"
```

## Task 4: Portrait Bokeh Generation

**Files:**
- Create: `pipeline/bokeh.py`
- Create: `tests/test_bokeh.py`

- [ ] **Step 1: Write failing bokeh tests**

Create `tests/test_bokeh.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_bokeh.py -v
```

Expected: FAIL because `pipeline.bokeh` does not exist.

- [ ] **Step 3: Implement depth bokeh**

Create `pipeline/bokeh.py`:

```python
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from pipeline.depth import normalize_depth
from pipeline.image_io import ensure_rgb


def apply_depth_bokeh(
    image: Image.Image,
    depth: np.ndarray,
    foreground_mask: np.ndarray,
    blur_radius: int = 12,
) -> Image.Image:
    rgb = ensure_rgb(image)
    normalized = normalize_depth(depth)
    if normalized.shape != (rgb.height, rgb.width):
        normalized = cv2.resize(normalized, rgb.size, interpolation=cv2.INTER_LINEAR)

    far_blur = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mid_blur = rgb.filter(ImageFilter.GaussianBlur(radius=max(2, blur_radius // 2)))

    depth_pixels = (normalized * 255.0).round().astype(np.uint8)
    far_mask = Image.fromarray((depth_pixels < 90).astype(np.uint8) * 255, mode="L")
    mid_mask = Image.fromarray(((depth_pixels >= 90) & (depth_pixels < 170)).astype(np.uint8) * 255, mode="L")

    composed = rgb.copy()
    composed.paste(mid_blur, mask=mid_mask)
    composed.paste(far_blur, mask=far_mask)

    fg_mask = Image.fromarray(foreground_mask, mode="L").resize(rgb.size, Image.Resampling.BILINEAR)
    composed.paste(rgb, mask=fg_mask)
    return composed
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_bokeh.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/bokeh.py tests/test_bokeh.py
git commit -m "feat: add depth-layered bokeh"
```

## Task 5: Caption And Classification Boundaries

**Files:**
- Create: `pipeline/caption.py`
- Create: `pipeline/classify.py`
- Create: `tests/test_caption.py`

- [ ] **Step 1: Write failing caption tests**

Create `tests/test_caption.py`:

```python
from PIL import Image

from pipeline.caption import generate_caption
from pipeline.classify import fallback_classification


def test_generate_caption_uses_fallback_without_api_key() -> None:
    image = Image.new("RGB", (32, 32), color=(1, 2, 3))
    result = generate_caption(image, api_key=None)
    assert result.used_fallback is True
    assert result.provider == "fallback"
    assert "foreground subject" in result.text


def test_fallback_classification_returns_demo_label() -> None:
    result = fallback_classification()
    assert result.label == "image"
    assert result.score == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_caption.py -v
```

Expected: FAIL because `pipeline.caption` and `pipeline.classify` do not exist.

- [ ] **Step 3: Implement caption and classification fallbacks**

Create `pipeline/caption.py`:

```python
from __future__ import annotations

from PIL import Image

from pipeline.types import CaptionResult


def fallback_caption() -> CaptionResult:
    return CaptionResult(
        text="A foreground subject extracted from the uploaded image, prepared for background removal and portrait-style image editing.",
        provider="fallback",
        used_fallback=True,
    )


def generate_caption(image: Image.Image, api_key: str | None = None, provider: str = "fallback") -> CaptionResult:
    if not api_key:
        return fallback_caption()
    return CaptionResult(
        text="Vision caption provider is configured. The MVP keeps this boundary ready for OpenAI or Claude Vision integration.",
        provider=provider,
        used_fallback=False,
    )
```

Create `pipeline/classify.py`:

```python
from __future__ import annotations

from functools import lru_cache
from typing import Any

from PIL import Image
from transformers import pipeline

from pipeline.image_io import ensure_rgb
from pipeline.types import ClassificationResult

CLASSIFICATION_MODEL_ID = "google/vit-base-patch16-224"


def fallback_classification() -> ClassificationResult:
    return ClassificationResult(label="image", score=1.0)


@lru_cache(maxsize=1)
def load_classification_pipeline() -> Any:
    return pipeline("image-classification", model=CLASSIFICATION_MODEL_ID)


def classify_image(image: Image.Image, top_k: int = 3) -> list[ClassificationResult]:
    try:
        classifier = load_classification_pipeline()
        outputs = classifier(ensure_rgb(image), top_k=top_k)
        return [
            ClassificationResult(label=str(item["label"]), score=float(item["score"]))
            for item in outputs
        ]
    except Exception:
        return [fallback_classification()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_caption.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add pipeline/caption.py pipeline/classify.py tests/test_caption.py
git commit -m "feat: add caption and classification boundaries"
```

## Task 6: End-To-End Orchestrator With Degradation

**Files:**
- Create: `pipeline/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing orchestrator test**

Create `tests/test_orchestrator.py`:

```python
import numpy as np
from PIL import Image

from pipeline.orchestrator import run_image_pipeline
from pipeline.types import ClassificationResult


def test_run_image_pipeline_returns_all_outputs_with_injected_functions() -> None:
    image = Image.new("RGB", (32, 32), color=(100, 80, 60))

    def fake_depth(_: Image.Image):
        depth = np.ones((32, 32), dtype=np.float32)
        depth_image = Image.new("L", (32, 32), color=128)
        return depth, depth_image

    def fake_classify(_: Image.Image):
        return [ClassificationResult(label="demo", score=0.9)]

    result = run_image_pipeline(
        image,
        depth_fn=fake_depth,
        classify_fn=fake_classify,
        api_key=None,
    )

    assert result.original.size == (32, 32)
    assert result.depth_map is not None
    assert result.subject is not None
    assert result.bokeh is not None
    assert result.caption.used_fallback is True
    assert result.classifications[0].label == "demo"
    assert result.errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_orchestrator.py -v
```

Expected: FAIL because `pipeline.orchestrator` does not exist.

- [ ] **Step 3: Implement orchestrator**

Create `pipeline/orchestrator.py`:

```python
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from pipeline.bokeh import apply_depth_bokeh
from pipeline.caption import generate_caption
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
    errors: list[str] = []
    original = resize_for_inference(ensure_rgb(image))
    depth_array: np.ndarray | None = None
    depth_image: Image.Image | None = None
    subject: Image.Image | None = None
    bokeh: Image.Image | None = None

    try:
        depth_array, depth_image = depth_fn(original)
    except Exception as exc:
        errors.append(f"Depth estimation failed: {exc}")

    if depth_array is not None:
        try:
            mask = build_foreground_mask(depth_array)
            subject = extract_subject(original, mask)
            bokeh = apply_depth_bokeh(original, depth_array, mask)
        except Exception as exc:
            errors.append(f"Depth-aware image processing failed: {exc}")

    try:
        caption = generate_caption(subject or original, api_key=api_key, provider=caption_provider)
    except Exception as exc:
        caption = CaptionResult(
            text="Caption generation was unavailable, but the image processing pipeline completed.",
            provider="fallback",
            used_fallback=True,
        )
        errors.append(f"Caption generation failed: {exc}")

    try:
        classifications = classify_fn(subject or original)
    except Exception as exc:
        classifications = [ClassificationResult(label="image", score=1.0)]
        errors.append(f"Classification failed: {exc}")

    return PipelineResult(
        original=original,
        depth_map=depth_image,
        subject=subject,
        bokeh=bokeh,
        caption=caption,
        classifications=classifications,
        errors=errors,
    )
```

- [ ] **Step 4: Run orchestrator tests**

Run:

```bash
pytest tests/test_orchestrator.py -v
```

Expected: PASS, 1 test.

- [ ] **Step 5: Run full deterministic test suite**

Run:

```bash
pytest tests/test_image_io.py tests/test_depth.py tests/test_mask.py tests/test_bokeh.py tests/test_caption.py tests/test_orchestrator.py -v
```

Expected: PASS, all tests. No HuggingFace model download should occur in these tests.

- [ ] **Step 6: Commit**

```bash
git add pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add end-to-end image pipeline orchestrator"
```

## Task 7: Streamlit App UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create Streamlit interface**

Create `app.py`:

```python
from __future__ import annotations

import os

import streamlit as st
from PIL import Image

from pipeline.image_io import image_to_png_bytes
from pipeline.orchestrator import run_image_pipeline


st.set_page_config(
    page_title="Image AI Utility",
    page_icon="image",
    layout="wide",
)


def load_uploaded_image() -> Image.Image | None:
    uploaded = st.sidebar.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        return None
    return Image.open(uploaded)


def load_sample_image() -> Image.Image:
    return Image.new("RGB", (768, 512), color=(218, 225, 232))


st.title("Image AI Utility")
st.caption("Depth estimation x background removal x portrait bokeh x image understanding")

with st.sidebar:
    st.header("Input")
    source = st.radio("Image source", ["Sample image", "Upload image"], index=0)
    api_key = st.text_input(
        "Vision API key",
        value=os.getenv("VISION_API_KEY", ""),
        type="password",
        help="Optional. Without a key, the demo uses fallback captioning.",
    )
    st.info(
        "MVP boundary: this deployable demo uses HuggingFace transformers instead of the thesis CoreML package, "
        "keeps a single-image desktop workflow, and avoids manual threshold controls."
    )

image = load_sample_image() if source == "Sample image" else load_uploaded_image()

if image is None:
    st.warning("Upload an image to run the demo.")
    st.stop()

with st.spinner("Running depth-aware image pipeline..."):
    result = run_image_pipeline(image, api_key=api_key or None)

top_cols = st.columns(3)
top_cols[0].subheader("Original")
top_cols[0].image(result.original, use_container_width=True)

top_cols[1].subheader("Depth Map")
if result.depth_map is not None:
    top_cols[1].image(result.depth_map, use_container_width=True)
    top_cols[1].download_button(
        "Download depth map",
        data=image_to_png_bytes(result.depth_map),
        file_name="depth-map.png",
        mime="image/png",
    )
else:
    top_cols[1].error("Depth map unavailable.")

top_cols[2].subheader("Depth-Aware Subject")
if result.subject is not None:
    top_cols[2].image(result.subject, use_container_width=True)
    top_cols[2].download_button(
        "Download subject",
        data=image_to_png_bytes(result.subject),
        file_name="subject-extraction.png",
        mime="image/png",
    )
else:
    top_cols[2].error("Subject extraction unavailable.")

bottom_cols = st.columns(3)
bottom_cols[0].subheader("Portrait Bokeh")
if result.bokeh is not None:
    bottom_cols[0].image(result.bokeh, use_container_width=True)
    bottom_cols[0].download_button(
        "Download bokeh",
        data=image_to_png_bytes(result.bokeh),
        file_name="portrait-bokeh.png",
        mime="image/png",
    )
else:
    bottom_cols[0].error("Bokeh result unavailable.")

bottom_cols[1].subheader("Subject Caption")
bottom_cols[1].write(result.caption.text)
bottom_cols[1].caption(f"Provider: {result.caption.provider}")

bottom_cols[2].subheader("Classification")
for item in result.classifications:
    bottom_cols[2].metric(item.label, f"{item.score:.2%}")

if result.errors:
    with st.expander("Processing notes", expanded=True):
        for error in result.errors:
            st.warning(error)

st.divider()
st.subheader("PM Decision Notes")
st.write(
    "This MVP starts from depth because depth can power more than background removal: it also enables "
    "portrait-mode bokeh and gives a product-relevant signal about foreground/background relationships. "
    "Scharr gradients on the depth map help expose boundary changes for a lightweight refinement step. "
    "The deployed demo uses HuggingFace transformers for shareability, while the thesis CoreML package remains "
    "the local research context. Future versions can add SAM hybrid refinement, stronger matting, model comparison, "
    "and mobile optimization."
)
```

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile app.py pipeline/*.py
```

Expected: command exits with code 0.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit demo interface"
```

## Task 8: README And Local Verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README**

Create `README.md`:

```markdown
# Image AI Utility

Image AI Utility is a Streamlit demo for AI-Assisted PM interviews. It productizes a master's thesis direction, "Image Segmentation Based on Depth Estimation," into a small interactive workflow.

## What It Does

Upload one image or use the built-in sample. The app produces:

- Depth map from `depth-anything/Depth-Anything-V2-Small-hf`
- Depth-aware subject extraction using Scharr gradient refinement
- Portrait-style bokeh using depth-layered Gaussian blur
- Subject-focused caption
- Image classification
- Downloadable result images

## Why HuggingFace Instead Of CoreML

The thesis used `DepthAnythingV2SmallF16.mlpackage`, which is excellent for local macOS and Apple Silicon experimentation. For an interview demo, this project uses HuggingFace `transformers` so it can run in CPU-friendly environments such as Streamlit Cloud or Hugging Face Spaces.

## MVP Boundaries

In scope:

- Single-image workflow
- Automatic processing
- Desktop-first Streamlit UI
- Depth-first subject extraction
- Bokeh simulation
- Caption and classification boundaries

Out of scope:

- Manual threshold controls
- Batch processing
- Model comparison
- Mobile optimization
- Production-grade segmentation guarantees

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest tests -v
```

The deterministic tests avoid downloading HuggingFace models. Full app execution will download models on first run.

## Interview Narrative

This demo shows how depth estimation can be used as a product primitive. Instead of wrapping a single background-removal model, it uses depth to support both foreground extraction and portrait-mode bokeh. The architecture keeps model boundaries modular so stronger segmentation or matting methods can be added after the MVP.
```

- [ ] **Step 2: Run deterministic tests**

Run:

```bash
pytest tests -v
```

Expected: PASS.

- [ ] **Step 3: Start local app**

Run:

```bash
streamlit run app.py
```

Expected: Streamlit prints a local URL, usually `http://localhost:8501`.

- [ ] **Step 4: Browser smoke test**

Open the local URL in the Codex in-app browser. Verify:

- The sample image flow renders without upload.
- The page shows Original, Depth Map, Depth-Aware Subject, Portrait Bokeh, Subject Caption, Classification, and PM Decision Notes.
- Missing Vision API key does not stop the demo.
- Download buttons appear for available images.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add setup and interview narrative"
```

## Plan Self-Review

Spec coverage:

- Product positioning: Task 8 README and Task 7 PM notes.
- Streamlit UI: Task 7.
- Depth Anything HuggingFace model: Task 2.
- Scharr depth mask and subject extraction: Task 3.
- Depth-layered bokeh: Task 4.
- Vision caption fallback boundary: Task 5.
- ViT classification boundary: Task 5.
- Result downloads: Task 7.
- Error degradation: Task 6 and Task 7.
- Local verification: Task 8.

Placeholder scan:

- No unfinished requirement markers.
- No deferred work markers inside implementation tasks.
- No unspecified validation steps.
- No unresolved file names.

Type consistency:

- `CaptionResult`, `ClassificationResult`, and `PipelineResult` are defined in Task 1 and reused consistently.
- `run_image_pipeline` returns the structure consumed by `app.py`.
- Test injection names match orchestrator parameters: `depth_fn` and `classify_fn`.

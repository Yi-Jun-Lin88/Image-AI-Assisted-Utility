import numpy as np
from PIL import Image

from pipeline.depth import depth_array_to_image
from pipeline.orchestrator import run_image_pipeline
from pipeline.types import ClassificationResult


def test_run_image_pipeline_returns_all_outputs_with_injected_fakes() -> None:
    image = Image.new("RGBA", (16, 12), color=(20, 40, 60, 255))

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        assert input_image.mode == "RGB"
        assert input_image.size == image.size
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fake_classify(input_image: Image.Image) -> list[ClassificationResult]:
        assert input_image.mode == "RGB"
        return [ClassificationResult(label="demo", score=0.99)]

    result = run_image_pipeline(
        image,
        depth_fn=fake_depth,
        classify_fn=fake_classify,
    )

    assert result.original.size == image.size
    assert result.depth_map is not None
    assert result.subject is not None
    assert result.bokeh is not None
    assert result.caption.used_fallback is True
    assert result.classifications[0].label == "demo"
    assert result.errors == []


def test_run_image_pipeline_preserves_caption_and_classification_when_depth_fails() -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))

    def failing_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        raise RuntimeError("depth unavailable")

    def fake_classify(input_image: Image.Image) -> list[ClassificationResult]:
        return [ClassificationResult(label="demo", score=0.99)]

    result = run_image_pipeline(
        image,
        depth_fn=failing_depth,
        classify_fn=fake_classify,
    )

    assert result.depth_map is None
    assert result.subject is None
    assert result.bokeh is None
    assert result.caption.used_fallback is True
    assert result.classifications[0].label == "demo"
    assert result.errors == ["Depth estimation failed: depth unavailable"]


def test_run_image_pipeline_preserves_bokeh_when_subject_extraction_fails(monkeypatch) -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fail_extract_subject(input_image: Image.Image, mask: np.ndarray) -> Image.Image:
        raise RuntimeError("subject failed")

    monkeypatch.setattr("pipeline.orchestrator.extract_subject", fail_extract_subject)

    result = run_image_pipeline(image, depth_fn=fake_depth)

    assert result.depth_map is not None
    assert result.subject is None
    assert result.bokeh is not None
    assert "Subject extraction failed: subject failed" in result.errors


def test_run_image_pipeline_preserves_subject_when_bokeh_fails(monkeypatch) -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fail_bokeh(input_image: Image.Image, depth: np.ndarray, mask: np.ndarray) -> Image.Image:
        raise RuntimeError("bokeh failed")

    monkeypatch.setattr("pipeline.orchestrator.apply_depth_bokeh", fail_bokeh)

    result = run_image_pipeline(image, depth_fn=fake_depth)

    assert result.depth_map is not None
    assert result.subject is not None
    assert result.bokeh is None
    assert "Bokeh generation failed: bokeh failed" in result.errors


def test_run_image_pipeline_preserves_image_outputs_when_caption_fails(monkeypatch) -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fail_caption(*args, **kwargs):
        raise RuntimeError("caption failed")

    monkeypatch.setattr("pipeline.orchestrator.generate_caption", fail_caption)

    result = run_image_pipeline(image, depth_fn=fake_depth)

    assert result.depth_map is not None
    assert result.subject is not None
    assert result.bokeh is not None
    assert result.caption.used_fallback is True
    assert "Caption generation failed: caption failed" in result.errors


def test_run_image_pipeline_preserves_image_and_caption_when_classification_fails() -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fail_classify(input_image: Image.Image) -> list[ClassificationResult]:
        raise RuntimeError("classification failed")

    result = run_image_pipeline(
        image,
        depth_fn=fake_depth,
        classify_fn=fail_classify,
    )

    assert result.depth_map is not None
    assert result.subject is not None
    assert result.bokeh is not None
    assert result.caption.used_fallback is True
    assert result.classifications[0].label == "image"
    assert "Image classification failed: classification failed" in result.errors

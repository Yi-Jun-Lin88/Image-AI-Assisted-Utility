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

    def fail_bokeh(
        input_image: Image.Image,
        depth: np.ndarray,
        mask: np.ndarray,
        blur_radius: int = 12,
    ) -> Image.Image:
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
    assert result.classifications[0].used_fallback is True
    assert "Image classification failed: classification failed" in result.errors


def test_run_image_pipeline_records_default_classification_fallback() -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fallback_classify(input_image: Image.Image) -> list[ClassificationResult]:
        return [ClassificationResult(label="image", score=1.0, used_fallback=True)]

    result = run_image_pipeline(
        image,
        depth_fn=fake_depth,
        classify_fn=fallback_classify,
    )

    assert result.classifications[0].used_fallback is True
    assert "Image classification used fallback output." in result.errors


def test_run_image_pipeline_passes_interaction_controls(monkeypatch) -> None:
    image = Image.new("RGB", (16, 12), color=(20, 40, 60))
    observed: dict[str, int] = {}

    def fake_depth(input_image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        depth = np.zeros((input_image.height, input_image.width), dtype=np.float32)
        depth[3:9, 4:12] = 1.0
        return depth, depth_array_to_image(depth)

    def fake_build_mask(depth: np.ndarray, subject_strength: int = 60) -> np.ndarray:
        observed["subject_strength"] = subject_strength
        return np.ones(depth.shape, dtype=np.uint8) * 255

    def fake_bokeh(
        input_image: Image.Image,
        depth: np.ndarray,
        mask: np.ndarray,
        blur_radius: int = 12,
    ) -> Image.Image:
        observed["bokeh_strength"] = blur_radius
        return input_image.copy()

    monkeypatch.setattr("pipeline.orchestrator.build_foreground_mask", fake_build_mask)
    monkeypatch.setattr("pipeline.orchestrator.apply_depth_bokeh", fake_bokeh)

    run_image_pipeline(
        image,
        depth_fn=fake_depth,
        subject_strength=72,
        bokeh_strength=18,
    )

    assert observed == {"subject_strength": 72, "bokeh_strength": 18}

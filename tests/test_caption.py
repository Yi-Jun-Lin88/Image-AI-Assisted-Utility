from PIL import Image
import pytest

from pipeline.caption import generate_caption
from pipeline.classify import classify_image, fallback_classification


def test_generate_caption_uses_fallback_without_api_key() -> None:
    image = Image.new("RGB", (8, 8), color="white")
    result = generate_caption(image)

    assert result.used_fallback is True
    assert result.provider == "fallback"
    assert "foreground subject" in result.text


def test_generate_caption_uses_fallback_with_api_key() -> None:
    image = Image.new("RGB", (8, 8), color="white")
    result = generate_caption(image, api_key="dummy")

    assert result.used_fallback is True
    assert result.provider == "fallback"


def test_fallback_classification_returns_image_label() -> None:
    result = fallback_classification()

    assert result.label == "image"
    assert result.score == 1.0
    assert result.used_fallback is True


def test_classify_image_falls_back_when_model_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_model_error():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("pipeline.classify.load_classification_pipeline", raise_model_error)
    image = Image.new("RGB", (8, 8), color="white")

    result = classify_image(image)

    assert len(result) == 1
    assert result[0].label == "image"
    assert result[0].score == 1.0
    assert result[0].used_fallback is True


def test_classify_image_falls_back_on_malformed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadClassifier:
        def __call__(self, image: Image.Image, top_k: int):
            return [{"label": "object"}]

    monkeypatch.setattr("pipeline.classify.load_classification_pipeline", lambda: BadClassifier())
    image = Image.new("RGB", (8, 8), color="white")

    result = classify_image(image)

    assert len(result) == 1
    assert result[0].label == "image"
    assert result[0].score == 1.0
    assert result[0].used_fallback is True

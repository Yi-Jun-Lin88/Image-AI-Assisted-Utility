from __future__ import annotations

from functools import lru_cache
from typing import Any

from PIL import Image

from pipeline.image_io import ensure_rgb
from pipeline.types import ClassificationResult

CLASSIFICATION_MODEL_ID = "google/vit-base-patch16-224"


def fallback_classification() -> ClassificationResult:
    return ClassificationResult(label="image", score=1.0, used_fallback=True)


@lru_cache(maxsize=1)
def load_classification_pipeline() -> Any:
    from transformers import pipeline

    return pipeline("image-classification", model=CLASSIFICATION_MODEL_ID)


def _parse_classification_outputs(outputs: Any) -> list[ClassificationResult]:
    if not isinstance(outputs, list):
        return [fallback_classification()]

    results: list[ClassificationResult] = []
    for output in outputs:
        if not isinstance(output, dict):
            return [fallback_classification()]
        if "label" not in output or "score" not in output:
            return [fallback_classification()]
        try:
            results.append(
                ClassificationResult(
                    label=str(output["label"]),
                    score=float(output["score"]),
                )
            )
        except (TypeError, ValueError):
            return [fallback_classification()]

    return results or [fallback_classification()]


def classify_image(image: Image.Image, top_k: int = 3) -> list[ClassificationResult]:
    try:
        classifier = load_classification_pipeline()
        outputs = classifier(ensure_rgb(image), top_k=top_k)
    except Exception:
        return [fallback_classification()]
    return _parse_classification_outputs(outputs)

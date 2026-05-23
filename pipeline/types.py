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

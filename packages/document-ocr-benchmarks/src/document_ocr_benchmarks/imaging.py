"""Image loading + deterministic quality assessment.

Quality is computed locally with OpenCV rather than asked of a VLM: it must be
cheap, deterministic, and provider-independent so it can gate recapture before
any model call. All providers share these helpers.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .models import QualityResult


@dataclass
class LoadedImage:
    """An image in memory plus lazily-derived representations."""

    path: Optional[Path]
    data: bytes
    content_type: str

    @classmethod
    def from_path(cls, path: str | Path) -> "LoadedImage":
        p = Path(path)
        data = p.read_bytes()
        ctype, _ = mimetypes.guess_type(str(p))
        return cls(path=p, data=data, content_type=ctype or "image/jpeg")

    @cached_property
    def pil(self) -> Image.Image:
        from io import BytesIO

        return Image.open(BytesIO(self.data)).convert("RGB")

    @cached_property
    def array(self) -> np.ndarray:
        """RGB uint8 ndarray."""
        return np.asarray(self.pil)

    @cached_property
    def gray(self) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.array, cv2.COLOR_RGB2GRAY)

    @property
    def size(self) -> tuple[int, int]:
        return self.pil.size  # (width, height)

    def data_url(self) -> str:
        b64 = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.content_type};base64,{b64}"


def assess_quality(image: LoadedImage) -> QualityResult:
    """Heuristic, deterministic quality signals (spec: Document Quality)."""
    import cv2

    gray = image.gray
    h, w = gray.shape[:2]

    # Blur: variance of the Laplacian. Higher = sharper.
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 60:
        blur = "high"
    elif lap_var < 200:
        blur = "medium"
    else:
        blur = "low"

    # Glare: fraction of near-saturated bright pixels.
    bright_frac = float(np.mean(gray > 245))
    if bright_frac > 0.12:
        glare = "high"
    elif bright_frac > 0.04:
        glare = "medium"
    else:
        glare = "low"

    # Underexposure (low light): mean luminance.
    mean_lum = float(np.mean(gray))
    low_light = mean_lum < 70

    # Cropped heuristic: strong content touching the image border suggests the
    # document was cut off.
    edges = cv2.Canny(gray, 50, 150)
    border = np.concatenate([
        edges[0, :], edges[-1, :], edges[:, 0], edges[:, -1]
    ])
    cropped = float(np.mean(border > 0)) > 0.25

    # Composite score in [0, 1].
    blur_score = min(lap_var / 300.0, 1.0)
    glare_score = max(0.0, 1.0 - bright_frac / 0.15)
    light_score = min(mean_lum / 130.0, 1.0)
    crop_score = 0.6 if cropped else 1.0
    quality_score = round(
        0.4 * blur_score + 0.25 * glare_score + 0.2 * light_score + 0.15 * crop_score,
        3,
    )

    readable = blur != "high" and not low_light and quality_score > 0.45

    return QualityResult(
        readable=readable,
        blur=blur,
        glare=glare,
        cropped=cropped,
        orientation="upright" if h >= w * 0.5 else "rotated",
        quality_score=quality_score,
    )

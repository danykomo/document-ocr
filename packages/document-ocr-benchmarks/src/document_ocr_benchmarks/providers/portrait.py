"""Document portrait extraction via face detection.

VLMs describe images; they do not crop them. So portrait extraction is a shared
deterministic step (spec fallback: "portrait extraction: qwen-vl -> face
detector crop"). Any provider can delegate here. Uses OpenCV's Haar cascade,
which needs no model download and runs on CPU.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

from ..imaging import LoadedImage
from ..models import PortraitResult


def _laplacian_sharpness(gray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_portrait(
    image: LoadedImage, *, source: str = "face-detector", return_image: bool = False
) -> PortraitResult:
    import cv2
    import numpy as np

    gray = image.gray
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return PortraitResult(available=False, source=source)

    # Largest detected face wins.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    rgb = image.array
    crop = rgb[y : y + h, x : x + w]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Quality: blend of sharpness and crop size relative to the document.
    sharp = min(_laplacian_sharpness(crop_gray) / 250.0, 1.0)
    img_h, img_w = gray.shape[:2]
    size_ratio = (w * h) / float(img_w * img_h)
    size_score = min(size_ratio / 0.10, 1.0)  # ~10% of doc area = full marks
    quality_score = round(0.6 * sharp + 0.4 * size_score, 3)
    # Detector confidence proxy: more neighbors implies a stronger hit, but
    # detectMultiScale here returns no scores, so derive from quality.
    confidence = round(0.5 + 0.5 * quality_score, 3)

    image_b64: Optional[str] = None
    if return_image:
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(crop).save(buf, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return PortraitResult(
        available=True,
        bbox=[int(x), int(y), int(x + w), int(y + h)],
        confidence=confidence,
        source=source,
        quality_score=quality_score,
        image_base64=image_b64,
    )

"""Capture-condition degradations.

Each function maps a clean PIL image to a degraded one, simulating the real
capture conditions the spec's benchmark groups call for (mobile, low-light,
glare, cropped, rotated, photocopy, WhatsApp-style compression, blur).
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..models import CaptureCondition


def blur(img: Image.Image, radius: float = 2.4) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def low_light(img: Image.Image, factor: float = 0.35) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def glare(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.62, h * 0.4
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    spot = np.clip(1.0 - dist / (0.45 * max(h, w)), 0, 1) ** 2
    arr += (spot[..., None] * 200.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def rotate(img: Image.Image, degrees: float = 9.0) -> Image.Image:
    return img.rotate(degrees, expand=True, fillcolor=(190, 190, 190))


def crop(img: Image.Image, frac: float = 0.12) -> Image.Image:
    w, h = img.size
    dx, dy = int(w * frac), int(h * frac)
    return img.crop((dx, dy, w - dx // 2, h))


def photocopy(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    arr = np.asarray(gray).astype(np.float32)
    arr = np.clip((arr - 110) * 1.8 + 128, 0, 255)
    noisy = arr + np.random.default_rng(7).normal(0, 12, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).convert("RGB")


def whatsapp(img: Image.Image, max_side: int = 900, quality: int = 28) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = BytesIO()
    small.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def mobile(img: Image.Image) -> Image.Image:
    # Mild combination: slight blur + slight perspective-ish brightness drop.
    out = blur(img, radius=1.1)
    return ImageEnhance.Brightness(out).enhance(0.9)


DEGRADATIONS = {
    CaptureCondition.CLEAN: lambda im: im,
    CaptureCondition.MOBILE: mobile,
    CaptureCondition.BLURRED: blur,
    CaptureCondition.LOW_LIGHT: low_light,
    CaptureCondition.GLARE: glare,
    CaptureCondition.ROTATED: rotate,
    CaptureCondition.CROPPED: crop,
    CaptureCondition.PHOTOCOPY: photocopy,
    CaptureCondition.WHATSAPP: whatsapp,
}


def apply(condition: CaptureCondition, img: Image.Image) -> Image.Image:
    return DEGRADATIONS.get(condition, lambda im: im)(img)

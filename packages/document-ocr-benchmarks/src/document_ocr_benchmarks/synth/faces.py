"""Procedural face generation for synthetic ID documents.

Real ID portraits cannot be used for synthetic specimens (privacy/legal). We
render a simple frontal "face" so the portrait-extraction step has a target and
the ground truth can record ``portrait_present=True`` with a bbox. Note: Haar
cascades are trained on photographs, so detection on drawn faces is imperfect —
portrait *accuracy* benchmarking should ultimately use real, legally-cleared
samples. The synthetic path validates the wiring and bbox bookkeeping.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw


def make_face(width: int = 150, height: int = 190, seed: int | None = None) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (210, 215, 225))
    draw = ImageDraw.Draw(img)

    skin = rng.choice([(196, 150, 110), (150, 110, 78), (120, 86, 60), (172, 130, 96)])
    cx, cy = width // 2, int(height * 0.52)
    fw, fh = int(width * 0.62), int(height * 0.7)

    # Hair backing.
    draw.ellipse(
        [cx - fw // 2 - 6, cy - fh // 2 - 14, cx + fw // 2 + 6, cy + fh // 2],
        fill=(40, 30, 28),
    )
    # Face oval.
    draw.ellipse(
        [cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2], fill=skin
    )
    # Eyes (dark regions create the light-dark pattern Haar keys on).
    eye_y = cy - fh // 8
    eye_dx = fw // 5
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse([ex - 11, eye_y - 7, ex + 11, eye_y + 7], fill=(245, 245, 245))
        draw.ellipse([ex - 5, eye_y - 5, ex + 5, eye_y + 5], fill=(35, 30, 30))
    # Eyebrows.
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.line([ex - 13, eye_y - 13, ex + 13, eye_y - 15], fill=(40, 30, 28), width=3)
    # Nose.
    draw.line([cx, eye_y + 6, cx - 5, cy + fh // 8], fill=tuple(max(0, c - 40) for c in skin), width=2)
    # Mouth.
    mouth_y = cy + fh // 4
    draw.arc([cx - 20, mouth_y - 8, cx + 20, mouth_y + 10], 200, 340, fill=(120, 60, 60), width=3)

    # Subtle shading so contrast resembles a photo a little more.
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(seed or 0).normal(0, 4, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

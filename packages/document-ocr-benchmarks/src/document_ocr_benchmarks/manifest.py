"""Load and validate manifests + expected-field ground truth.

Supports both synthetic datasets and dropped-in real samples, as long as they
follow the same on-disk layout:

    <root>/manifests/<name>.json   # Manifest
    <root>/samples/<id>.<ext>      # images referenced by image_path
    <root>/expected/<id>.json      # ExpectedFields per sample
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import ExpectedFields, Manifest, Sample


def load_manifest(path: str | Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    return Manifest.model_validate(data)


def load_expected(expected_dir: str | Path, sample_id: str) -> Optional[ExpectedFields]:
    p = Path(expected_dir) / f"{sample_id}.json"
    if not p.exists():
        return None
    return ExpectedFields.model_validate_json(p.read_text())


def validate_dataset(manifest: Manifest, root: str | Path) -> list[str]:
    """Return a list of human-readable problems (empty == valid)."""
    root = Path(root)
    problems: list[str] = []
    seen: set[str] = set()
    for s in manifest.samples:
        if s.sample_id in seen:
            problems.append(f"duplicate sample_id: {s.sample_id}")
        seen.add(s.sample_id)
        if not (root / s.image_path).exists():
            problems.append(f"{s.sample_id}: missing image {s.image_path}")
        if load_expected(root / "expected", s.sample_id) is None:
            problems.append(f"{s.sample_id}: missing expected/{s.sample_id}.json")
    return problems


def resolve_image_path(root: str | Path, sample: Sample) -> Path:
    return Path(root) / sample.image_path

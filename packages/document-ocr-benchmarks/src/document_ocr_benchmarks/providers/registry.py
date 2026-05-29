"""Provider registry + factory.

Maps provider names to runner factories so the harness can instantiate any
candidate by name. Metadata (license, cost, deployment notes) is available from
the classes even when a provider is not configured/installed, which lets the
report describe every candidate's commercial posture regardless of runtime.
"""

from __future__ import annotations

from typing import Callable

from ..config import Settings
from .base import ProviderRunner
from .glm_ocr import GLMOCRRunner
from .mock import MockRunner
from .olmocr import OlmOCRRunner
from .paddleocr_vl import PaddleOCRVLRunner
from .qwen_vl import QwenVLRunner
from .tesseract import TesseractRunner

# Name -> class (for static metadata listing).
PROVIDER_CLASSES: dict[str, type[ProviderRunner]] = {
    QwenVLRunner.name: QwenVLRunner,
    GLMOCRRunner.name: GLMOCRRunner,
    PaddleOCRVLRunner.name: PaddleOCRVLRunner,
    OlmOCRRunner.name: OlmOCRRunner,
    TesseractRunner.name: TesseractRunner,
    MockRunner.name: MockRunner,
}


def build_provider(name: str, settings: Settings) -> ProviderRunner:
    """Instantiate a provider runner wired with its configuration."""
    name = name.strip().lower()
    if name in {"qwen-vl", "glm-ocr", "paddleocr-vl", "olmocr"}:
        ep = settings.endpoint_for(name)
        cls = PROVIDER_CLASSES[name]
        return cls(endpoint=ep, timeout_ms=settings.sync_timeout_ms)  # type: ignore[call-arg]
    if name == "tesseract":
        return TesseractRunner(lang=settings.tesseract_lang)
    if name == "mock":
        return MockRunner()
    raise KeyError(f"Unknown provider: {name}")


def make_factory(name: str, settings: Settings) -> Callable[[], ProviderRunner]:
    return lambda: build_provider(name, settings)


def known_providers() -> list[str]:
    return list(PROVIDER_CLASSES.keys())

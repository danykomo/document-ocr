"""PaddleOCR-VL runner (efficient production baseline contender).

Served via vLLM as an OpenAI-compatible endpoint (officially supported), so it
reuses the shared VLM backend.
"""

from __future__ import annotations

from ..models import CostEstimate, LicenseStatus
from .vlm_openai import OpenAICompatibleVLM


class PaddleOCRVLRunner(OpenAICompatibleVLM):
    name = "paddleocr-vl"
    display_name = "PaddleOCR-VL"
    license_name = "Apache-2.0"
    license_status = LicenseStatus.APPROVED
    cost_estimate = CostEstimate.LOW
    on_prem = True
    deployment_notes = (
        "0.9B (NaViT encoder + ERNIE-4.5-0.3B), Apache-2.0. 109 languages, strong "
        "layout/table/formula parsing, low serving cost; vLLM-supported. Main risk: "
        "field-level extraction quality on Nigerian IDs and bank forms vs general VLMs."
    )
    field_system_hint = (
        "You are PaddleOCR-VL extracting structured fields from Nigerian documents. "
        "Transcribe exact values, return null when a field is absent."
    )

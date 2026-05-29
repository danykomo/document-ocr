"""olmOCR runner (open, reproducible reference pipeline contender)."""

from __future__ import annotations

from ..models import CostEstimate, LicenseStatus
from .vlm_openai import OpenAICompatibleVLM


class OlmOCRRunner(OpenAICompatibleVLM):
    name = "olmocr"
    display_name = "olmOCR 2"
    license_name = "Apache-2.0"
    license_status = LicenseStatus.APPROVED
    # 7B (fine-tuned Qwen2.5-VL-7B): medium cost, English-print focused.
    cost_estimate = CostEstimate.MEDIUM
    on_prem = True
    deployment_notes = (
        "7B VLM fine-tuned from Qwen2.5-VL-7B, Apache-2.0. Reproducible reference "
        "pipeline (Ai2), strong on English print/PDF and tables/equations. Risks for "
        "this use case: tuned for document linearization, so field extraction and "
        "portrait extraction need extra components; less proven on Nigerian IDs."
    )
    field_system_hint = (
        "You extract structured fields from Nigerian identity and banking documents. "
        "Return exact transcribed values; use null for absent fields."
    )

"""Qwen-VL / Qwen3-VL runner (high-accuracy general VLM contender)."""

from __future__ import annotations

from ..models import CostEstimate, LicenseStatus
from .vlm_openai import OpenAICompatibleVLM


class QwenVLRunner(OpenAICompatibleVLM):
    name = "qwen-vl"
    display_name = "Qwen3-VL"
    license_name = "Apache-2.0"
    license_status = LicenseStatus.APPROVED
    # Heavier general VLM: strong accuracy, higher GPU/cost than 0.9B OCR models.
    cost_estimate = CostEstimate.HIGH
    on_prem = True
    deployment_notes = (
        "Apache-2.0. Sizes 2B/4B/8B/32B + MoE (30B-A3B, 235B-A22B). "
        "Serve via vLLM>=0.11 or SGLang, OpenAI-compatible. Strong document "
        "understanding and flexible JSON extraction; main risks are GPU cost and "
        "deterministic field output. 8B is the suggested on-prem default."
    )
    field_system_hint = (
        "You are Qwen3-VL extracting structured fields from Nigerian identity and "
        "banking documents. Be precise, never hallucinate values, prefer null."
    )

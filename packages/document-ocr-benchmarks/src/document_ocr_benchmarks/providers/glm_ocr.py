"""GLM-OCR runner (compact OCR-specialized contender)."""

from __future__ import annotations

from ..models import CostEstimate, LicenseStatus
from .vlm_openai import OpenAICompatibleVLM


class GLMOCRRunner(OpenAICompatibleVLM):
    name = "glm-ocr"
    display_name = "GLM-OCR"
    license_name = "MIT (model); PP-DocLayoutV3 component Apache-2.0"
    license_status = LicenseStatus.APPROVED
    # 0.9B model -> low GPU footprint, low cost. OCR/KIE specialized.
    cost_estimate = CostEstimate.LOW
    on_prem = True
    deployment_notes = (
        "0.9B (CogViT encoder + GLM-0.5B decoder), MIT-licensed model weights; the "
        "full pipeline bundles PP-DocLayoutV3 (Apache-2.0). Serve via vLLM, SGLang, "
        "or Ollama; edge-deployable. Strong document parsing + KIE for its size. "
        "Validate portrait behavior (OCR model, not a cropper) and runtime maturity."
    )
    field_system_hint = (
        "You are GLM-OCR performing key information extraction on Nigerian "
        "identity and banking documents. Return exact transcribed values, null if absent."
    )

"""PaddleOCR-VL runner.

PaddleOCR-VL is a *document parsing* model, not an instruction-following JSON
extractor: asked for JSON it returns its native transcribed text instead. We
play to its strength — ask it for a clean text transcription of the document,
then parse labeled fields out of that text. This makes the benchmark a fair
"PaddleOCR-VL as a higher-quality OCR than Tesseract" test rather than
penalising it for not following a prompt it was never trained for.
"""

from __future__ import annotations

from ..imaging import LoadedImage
from ..models import (
    CostEstimate,
    FieldExtractionResult,
    LicenseStatus,
    TextExtractionResult,
)
from ..schemas import DocumentSchema
from .text_kie import extract_fields_from_text
from .vlm_openai import OpenAICompatibleVLM


_TEXT_PROMPT = (
    "Read this Nigerian document and transcribe every visible field. "
    "Output each label and its value on their own line "
    "(format: 'Label:\\nValue'). Do not summarise, translate, or skip fields. "
    "Output ONLY the transcribed text, no commentary."
)


class PaddleOCRVLRunner(OpenAICompatibleVLM):
    name = "paddleocr-vl"
    display_name = "PaddleOCR-VL"
    license_name = "Apache-2.0"
    license_status = LicenseStatus.APPROVED
    cost_estimate = CostEstimate.LOW
    on_prem = True
    deployment_notes = (
        "0.9B (NaViT encoder + ERNIE-4.5-0.3B), Apache-2.0. 109 languages, strong "
        "layout/table/formula parsing, low serving cost; vLLM-supported. Document "
        "parser by training: its native output is transcribed text with layout "
        "preserved, not JSON. The harness uses its OCR output and parses labeled "
        "fields out of it — fair to its actual capability."
    )
    field_system_hint = (
        "You are PaddleOCR-VL, a precise document text transcriber for Nigerian "
        "identity and banking documents."
    )

    def _document_text(self, image: LoadedImage, max_tokens: int = 2048) -> str:
        return self._chat(self.field_system_hint, _TEXT_PROMPT, image, max_tokens=max_tokens)

    def extract_text(self, image: LoadedImage) -> TextExtractionResult:
        return TextExtractionResult(raw=self._document_text(image, max_tokens=4096).strip())

    def extract_fields(
        self, image: LoadedImage, schema: DocumentSchema
    ) -> FieldExtractionResult:
        text = self._document_text(image)
        return extract_fields_from_text(text, schema, source=self.name)

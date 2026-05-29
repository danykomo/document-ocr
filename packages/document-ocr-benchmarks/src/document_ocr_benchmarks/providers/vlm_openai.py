"""OpenAI-compatible VLM runner.

vLLM and SGLang both expose an OpenAI ``/v1/chat/completions`` endpoint that
accepts image content. Qwen-VL, GLM-OCR, PaddleOCR-VL, and olmOCR all serve
this way, so they share this backend and differ only in prompts / metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..config import ProviderEndpoint
from ..imaging import LoadedImage
from ..models import (
    Capability,
    ClassificationResult,
    DocumentType,
    FieldExtractionResult,
    FieldValue,
    PortraitResult,
    TextExtractionResult,
)
from ..normalization import normalize_field
from ..schemas import DocumentSchema
from .base import ProviderRunner
from .portrait import extract_portrait as detect_portrait

_DOC_TYPES = [d.value for d in DocumentType]

_CLASSIFY_PROMPT = (
    "You are a document classifier for Nigerian banking and identity documents. "
    "Identify the document type. Respond with ONLY a JSON object: "
    '{"documentType": "<one of: ' + ", ".join(_DOC_TYPES) + '>", '
    '"confidence": <0..1>}. No prose.'
)

_TEXT_PROMPT = (
    "Transcribe ALL text visible in this document image exactly as it appears, "
    "preserving reading order. Return plain text only, no commentary."
)


def _fields_prompt(schema: DocumentSchema) -> str:
    field_list = ", ".join(schema.field_names)
    return (
        "Extract the following fields from this Nigerian document image. "
        f"Fields: {field_list}. "
        "Return ONLY a JSON object mapping each field name to an object "
        '{"value": "<string or null>", "confidence": <0..1>}. '
        "Use null when a field is not present. Do not invent values. No prose."
    )


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response (tolerant of fences)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class OpenAICompatibleVLM(ProviderRunner):
    """Generic runner against an OpenAI-compatible chat-completions endpoint."""

    name = "openai-vlm"
    display_name = "OpenAI-compatible VLM"
    # Subclasses tune the field-extraction system hint.
    field_system_hint = "You extract structured data from document images precisely."

    def __init__(self, endpoint: ProviderEndpoint, timeout_ms: int = 60_000):
        self._ep = endpoint
        self._timeout = timeout_ms / 1000.0
        self.model_version = endpoint.model

    # -- availability ----------------------------------------------------- #
    def capabilities(self) -> set[Capability]:
        return {
            Capability.CLASSIFY,
            Capability.EXTRACT_TEXT,
            Capability.EXTRACT_FIELDS,
            Capability.EXTRACT_PORTRAIT,
            Capability.ASSESS_QUALITY,
        }

    def is_available(self) -> tuple[bool, Optional[str]]:
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False, "httpx not installed"
        if not self._ep.endpoint:
            return False, f"endpoint not configured (set OCR_PROVIDER_*_ENDPOINT)"
        return True, None

    # -- low-level chat call ---------------------------------------------- #
    def _chat(self, system: str, user_text: str, image: LoadedImage,
              max_tokens: int = 2048, temperature: float = 0.0) -> str:
        import httpx

        url = self._ep.endpoint.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._ep.api_key:
            headers["Authorization"] = f"Bearer {self._ep.api_key}"
        payload: dict[str, Any] = {
            "model": self._ep.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image.data_url()},
                        },
                    ],
                },
            ],
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    # -- operations ------------------------------------------------------- #
    def classify(self, image: LoadedImage) -> ClassificationResult:
        out = self._chat(_CLASSIFY_PROMPT, "Classify this document.", image, max_tokens=128)
        parsed = _extract_json(out) or {}
        raw_type = str(parsed.get("documentType", "unknown")).strip().lower()
        try:
            dtype = DocumentType(raw_type)
        except ValueError:
            dtype = DocumentType.UNKNOWN
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        return ClassificationResult(document_type=dtype, label=raw_type, confidence=conf)

    def extract_text(self, image: LoadedImage) -> TextExtractionResult:
        out = self._chat(
            "You are a precise OCR transcription engine.",
            _TEXT_PROMPT,
            image,
            max_tokens=4096,
        )
        return TextExtractionResult(raw=out.strip())

    def extract_fields(
        self, image: LoadedImage, schema: DocumentSchema
    ) -> FieldExtractionResult:
        out = self._chat(self.field_system_hint, _fields_prompt(schema), image)
        parsed = _extract_json(out)
        if parsed is None:
            return FieldExtractionResult(schema_followed=False)
        fields: dict[str, FieldValue] = {}
        for fname in schema.field_names:
            entry = parsed.get(fname)
            if entry is None:
                continue
            if isinstance(entry, dict):
                value = entry.get("value")
                conf = float(entry.get("confidence", 0.0) or 0.0)
            else:
                value = entry
                conf = 0.0
            if value in (None, "", "null"):
                continue
            value = str(value)
            fields[fname] = FieldValue(
                value=value,
                normalized_value=normalize_field(fname, value),
                confidence=conf,
                source=self.name,
            )
        # Schema-followed = valid JSON whose keys are within the requested schema.
        return FieldExtractionResult(fields=fields, schema_followed=True)

    def extract_portrait(self, image: LoadedImage) -> PortraitResult:
        # Delegate to the shared face detector and record provenance.
        return detect_portrait(image, source=f"{self.name}+face-detector")

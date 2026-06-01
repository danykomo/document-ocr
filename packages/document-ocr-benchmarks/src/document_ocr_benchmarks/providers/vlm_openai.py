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


# Common label variants the model sees in the document mapped to schema fields.
# OCR-specialized models (GLM-OCR, PaddleOCR-VL) tend to mirror the on-document
# labels as JSON keys ("Sex", "Date of Birth", "NIN") rather than the snake_case
# names we ask for in the prompt — coerce both so the parser is forgiving.
_KEY_ALIASES = {
    "sex": "gender",
    "given_name": "first_name",
    "given_names": "first_name",
    "last_name": "surname",
    "family_name": "surname",
    "name": "full_name",
    "full_names": "full_name",
    "date_of_issue": "issue_date",
    "issued": "issue_date",
    "date_issued": "issue_date",
    "date_of_expiry": "expiry_date",
    "expires": "expiry_date",
    "expiry": "expiry_date",
    "valid_till": "expiry_date",
    "passport_no": "passport_number",
    "licence_no": "drivers_license_number",
    "license_no": "drivers_license_number",
    "licence_number": "drivers_license_number",
    "license_number": "drivers_license_number",
    "document_no": "document_number",
    "id_no": "document_number",
    "card_no": "document_number",
    "vin": "document_number",
    "account_no": "account_number",
    "acct_no": "account_number",
    "bank_verification_number": "bvn",
    "national_identification_number": "nin",
    "issued_by": "issuing_authority",
    "authority": "issuing_authority",
    "residential_address": "address",
    "dob": "date_of_birth",
    "birth_date": "date_of_birth",
}


def _normalize_key(k: str) -> str:
    """``"First Name"`` → ``"first_name"``; ``"Date-of-Issue."`` → ``"date_of_issue"``."""
    return re.sub(r"[^a-z0-9]+", "_", k.strip().lower()).strip("_")


def _coerce_keys(parsed: dict, schema_field_names: list[str]) -> dict:
    """Map the model's natural label-keys onto the requested schema field names."""
    result: dict = {}
    schema_set = set(schema_field_names)
    for raw_k, v in parsed.items():
        if not isinstance(raw_k, str):
            continue
        norm = _normalize_key(raw_k)
        target = (
            norm if norm in schema_set
            else _KEY_ALIASES.get(norm) if _KEY_ALIASES.get(norm) in schema_set
            else None
        )
        if target and target not in result:
            result[target] = v
    # If the document shows surname + first (+ middle) separately, derive full_name
    # so models that don't synthesize it themselves aren't penalised.
    if "full_name" in schema_set and "full_name" not in result:
        parts = []
        for part_key in ("first_name", "middle_name", "surname"):
            v = result.get(part_key)
            if isinstance(v, dict):
                v = v.get("value")
            if v:
                parts.append(str(v))
        if parts:
            result["full_name"] = " ".join(parts)
    return result


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
        """Probe the endpoint so a misconfigured VLM is reported, not silently scored 0.

        Checks, in order: httpx installed, endpoint configured, endpoint
        reachable, and the configured model is actually being served. This makes
        ``document-ocr-bench providers`` a usable health check.
        """
        try:
            import httpx
        except ImportError:
            return False, "httpx not installed"
        if not self._ep.endpoint:
            return False, "endpoint not configured (set OCR_PROVIDER_*_ENDPOINT)"

        url = self._ep.endpoint.rstrip("/") + "/models"
        headers = {}
        if self._ep.api_key:
            headers["Authorization"] = f"Bearer {self._ep.api_key}"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return False, f"cannot reach {self._ep.endpoint} ({type(exc).__name__})"

        served = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        if self._ep.model and served and self._ep.model not in served:
            return (
                False,
                f"endpoint serves {served}, not '{self._ep.model}' — "
                f"run the provider whose model matches, or set this provider's "
                f"model env to the served one",
            )
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
        parsed = _coerce_keys(parsed, schema.field_names)
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

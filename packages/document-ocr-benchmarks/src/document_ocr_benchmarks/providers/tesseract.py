"""Tesseract classical OCR runner (fallback baseline).

CPU-only, no model download, always available once the system ``tesseract``
binary is installed. Represents the floor the VLM candidates must beat. Field
extraction is keyword/regex over the raw OCR text, which deliberately exposes
the weakness the spec calls out: "weak layout reasoning and structured field
extraction".
"""

from __future__ import annotations

import re
from typing import Optional

from ..imaging import LoadedImage
from ..models import (
    Capability,
    ClassificationResult,
    CostEstimate,
    DocumentType,
    FieldExtractionResult,
    FieldValue,
    LicenseStatus,
    PortraitResult,
    TextBlock,
    TextExtractionResult,
)
from ..normalization import normalize_field
from ..schemas import DocumentSchema
from .base import ProviderRunner
from .portrait import extract_portrait as detect_portrait

# Keyword fingerprints used for naive document-type classification.
_CLASSIFY_KEYWORDS: list[tuple[DocumentType, tuple[str, ...]]] = [
    (DocumentType.NIN_SLIP, ("national identification number", "nin slip", "nimc")),
    (DocumentType.NATIONAL_ID, ("national identity card", "national id")),
    (DocumentType.PASSPORT, ("passport", "federal republic of nigeria passport")),
    (DocumentType.DRIVERS_LICENSE, ("driver", "driving licence", "frsc")),
    (DocumentType.VOTER_ID, ("voter", "independent national electoral", "inec")),
    (DocumentType.BANK_MANDATE, ("mandate", "account opening", "bvn")),
    (DocumentType.UTILITY_BILL, ("electricity", "utility", "disco", "bill")),
    (DocumentType.BANK_STATEMENT, ("statement of account", "bank statement", "closing balance")),
]

# Field label aliases for keyword extraction.
_LABELS: dict[str, tuple[str, ...]] = {
    "surname": ("surname", "last name"),
    "first_name": ("first name", "given name", "given names"),
    "middle_name": ("middle name",),
    "full_name": ("name",),
    "date_of_birth": ("date of birth", "dob", "birth date"),
    "gender": ("sex", "gender"),
    "nin": ("nin", "national identification number"),
    "bvn": ("bvn", "bank verification number"),
    "passport_number": ("passport no", "passport number"),
    "drivers_license_number": ("licence no", "license no", "licence number"),
    "document_number": ("document no", "id no", "card no", "vin"),
    "account_number": ("account no", "account number", "acct no"),
    "issue_date": ("issue date", "date of issue", "issued"),
    "expiry_date": ("expiry", "date of expiry", "expires", "valid till"),
    "address": ("address", "residential address"),
    "issuing_authority": ("authority", "issued by"),
}


class TesseractRunner(ProviderRunner):
    name = "tesseract"
    display_name = "Tesseract (classical OCR)"
    license_name = "Apache-2.0"
    license_status = LicenseStatus.APPROVED
    cost_estimate = CostEstimate.LOW
    on_prem = True
    deployment_notes = (
        "Apache-2.0, CPU-only, fully local. Cheap text extraction and the safe "
        "local-only floor. Weak at layout reasoning and structured field "
        "extraction; use as fallback, not as primary field extractor."
    )

    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self._last_text: Optional[str] = None

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
            import pytesseract
        except ImportError:
            return False, "pytesseract not installed (pip install '.[tesseract]')"
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False, "tesseract binary not found on PATH"
        return True, None

    # -- ops -------------------------------------------------------------- #
    def extract_text(self, image: LoadedImage) -> TextExtractionResult:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            image.pil, lang=self.lang, output_type=Output.DICT
        )
        blocks: list[TextBlock] = []
        words: list[str] = []
        n = len(data["text"])
        for i in range(n):
            word = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else -1.0
            if not word or conf < 0:
                continue
            words.append(word)
            blocks.append(
                TextBlock(
                    text=word,
                    confidence=round(conf / 100.0, 3),
                    bbox=[
                        int(data["left"][i]),
                        int(data["top"][i]),
                        int(data["left"][i] + data["width"][i]),
                        int(data["top"][i] + data["height"][i]),
                    ],
                )
            )
        raw = pytesseract.image_to_string(image.pil, lang=self.lang)
        self._last_text = raw
        return TextExtractionResult(raw=raw.strip(), blocks=blocks)

    def classify(self, image: LoadedImage) -> ClassificationResult:
        text = (self._last_text or self.extract_text(image).raw).lower()
        best: tuple[DocumentType, int] = (DocumentType.UNKNOWN, 0)
        for dtype, keywords in _CLASSIFY_KEYWORDS:
            hits = sum(1 for kw in keywords if kw in text)
            if hits > best[1]:
                best = (dtype, hits)
        dtype, hits = best
        confidence = min(0.4 + 0.2 * hits, 0.9) if hits else 0.0
        return ClassificationResult(document_type=dtype, label=dtype.value, confidence=confidence)

    def extract_fields(
        self, image: LoadedImage, schema: DocumentSchema
    ) -> FieldExtractionResult:
        raw = self._last_text or self.extract_text(image).raw
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        fields: dict[str, FieldValue] = {}

        for fname in schema.field_names:
            value = self._find_labeled_value(fname, lines, raw)
            if value:
                fields[fname] = FieldValue(
                    value=value,
                    normalized_value=normalize_field(fname, value),
                    confidence=0.4,  # classical OCR: modest confidence
                    source=self.name,
                )
        # Classical OCR doesn't emit JSON, so "schema_followed" is False by design.
        return FieldExtractionResult(fields=fields, schema_followed=False)

    def extract_portrait(self, image: LoadedImage) -> PortraitResult:
        return detect_portrait(image, source="tesseract+face-detector")

    # -- helpers ---------------------------------------------------------- #
    def _find_labeled_value(self, field: str, lines: list[str], raw: str) -> Optional[str]:
        labels = _LABELS.get(field, ())
        # Special-cased pattern extraction for identity numbers.
        if field == "nin":
            m = re.search(r"\b(\d{11})\b", raw)
            return m.group(1) if m else None
        if field == "bvn":
            m = re.search(r"bvn[^0-9]{0,12}(\d{11})", raw, re.IGNORECASE)
            return m.group(1) if m else None
        if field == "passport_number":
            m = re.search(r"\b([A-Z]\d{8})\b", raw)
            return m.group(1) if m else None

        for line in lines:
            low = line.lower()
            for label in labels:
                if label in low:
                    # Take text after the label / colon.
                    idx = low.find(label) + len(label)
                    tail = line[idx:].lstrip(" :.-\t")
                    if tail:
                        return tail.strip()
        return None

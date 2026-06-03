"""Heuristic field extraction from OCR-style transcribed text.

Used by providers that return transcribed text rather than structured JSON
(PaddleOCR-VL by design, and as a fallback for any VLM whose JSON path fails).
Handles both ``Label: Value`` on the same line and ``Label:\\nValue`` on the
next line — PaddleOCR-VL emits the latter when reading column layouts.
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import FieldExtractionResult, FieldValue
from ..normalization import normalize_field
from ..schemas import DocumentSchema


# Mapping of schema field name → label strings that appear on Nigerian documents.
_LABELS_BY_FIELD: dict[str, tuple[str, ...]] = {
    # Names
    "surname": ("surname", "last name", "family name"),
    "first_name": ("first name", "given name", "given names", "firstname", "firstnames"),
    "middle_name": ("middle name", "middle names", "other names"),
    "full_name": ("name", "full name", "full names", "account name", "customer name"),
    "company_name": ("company name", "registered name", "name of company"),
    # Demographics
    "date_of_birth": ("date of birth", "dob", "birth date"),
    "gender": ("sex", "gender"),
    "nationality": ("nationality", "country"),
    "place_of_birth": ("place of birth", "pob"),
    "occupation": ("occupation", "profession", "employment"),
    "blood_group": ("blood group", "blood type"),
    "height": ("height",),
    # Identity numbers
    "nin": ("nin", "national identification number", "national id number"),
    "tracking_id": ("tracking id", "tracking", "tracking number"),
    "bvn": ("bvn", "bank verification number"),
    "passport_number": ("passport no", "passport number"),
    "drivers_license_number": (
        "licence no", "license no", "licence number", "license number",
        "dl no", "dl number",
    ),
    "document_number": (
        "document no", "id no", "card no", "vin", "voter id",
        "voter identification number",
    ),
    "account_number": ("account no", "account number", "acct no", "nuban"),
    "meter_number": ("meter no", "meter number", "meter id"),
    "registration_number": ("registration no", "registration number", "rc number",
                            "rc no", "bn number", "bn no", "rc/bc number"),
    "polling_unit_code": ("polling unit", "pu code", "polling unit code"),
    # Dates
    "issue_date": ("issue date", "date of issue", "issued", "date issued"),
    "expiry_date": ("expiry", "expiry date", "date of expiry", "expires", "valid till"),
    "due_date": ("due date", "payment due", "pay by"),
    "registration_date": ("registration date", "date of registration", "date registered"),
    "incorporation_date": ("date of incorporation", "incorporated on",
                           "date of registration", "registered on"),
    # Address / authority / period
    "address": ("address", "residential address", "service address", "registered address"),
    "issuing_authority": ("authority", "issued by", "issuing authority", "bank"),
    "issuing_state": ("state of issue", "issuing state", "state"),
    "branch_name": ("branch", "branch name"),
    "billing_period": ("billing period", "bill period", "period"),
    "statement_period": ("statement period", "period", "for the period"),
    # Money
    "amount_due": ("amount due", "total due", "amount payable", "balance due"),
    "opening_balance": ("opening balance", "balance brought forward", "b/f"),
    "closing_balance": ("closing balance", "balance carried forward", "c/f"),
    # Contact
    "phone_number": ("phone", "telephone", "mobile", "phone no", "telephone no"),
}


def _norm_label(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", s.lower())).strip()


def _build_label_map(schema: DocumentSchema) -> dict[str, str]:
    """Map every accepted label phrase → schema field name (restricted to schema)."""
    out: dict[str, str] = {}
    for fname in schema.field_names:
        for label in _LABELS_BY_FIELD.get(fname, ()):
            out[_norm_label(label)] = fname
        # Also accept the field name itself (e.g. "first_name" -> "first name")
        out[_norm_label(fname.replace("_", " "))] = fname
    return out


def _label_only_line(line: str, label_map: dict[str, str]) -> Optional[str]:
    """If the entire line is just a label (with optional trailing colon),
    return the schema field; otherwise ``None``."""
    candidate = line.rstrip(":").strip()
    norm = _norm_label(candidate)
    return label_map.get(norm) if norm in label_map else None


def _split_inline(line: str, label_map: dict[str, str]) -> Optional[tuple[str, str]]:
    """If the line is ``Label: Value``, return ``(schema_field, value)``."""
    m = re.match(r"^\s*([^:]{1,40}?):\s*(.+\S)\s*$", line)
    if not m:
        return None
    label_text = _norm_label(m.group(1))
    value = m.group(2).strip()
    if value and label_text in label_map:
        return label_map[label_text], value
    return None


def extract_fields_from_text(
    text: str, schema: DocumentSchema, source: str, confidence: float = 0.5
) -> FieldExtractionResult:
    """Pull labeled fields out of OCR text into a ``FieldExtractionResult``.

    The keys are guaranteed to be schema field names (snake_case). Unknown labels
    are skipped; the first match for each schema field wins.
    """
    if not text:
        return FieldExtractionResult(schema_followed=False)

    label_map = _build_label_map(schema)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fields: dict[str, FieldValue] = {}

    def add(field: str, value: str) -> None:
        if not value or field in fields:
            return
        fields[field] = FieldValue(
            value=value,
            normalized_value=normalize_field(field, value),
            confidence=confidence,
            source=source,
        )

    for i, line in enumerate(lines):
        # 1. Inline "Label: Value"
        inline = _split_inline(line, label_map)
        if inline:
            add(*inline)
            continue
        # 2. Standalone "Label:" — the value is the next non-label line.
        field = _label_only_line(line, label_map)
        if field is None:
            continue
        for next_line in lines[i + 1 :]:
            if _label_only_line(next_line, label_map) is not None:
                break  # next label without a value in between
            if _split_inline(next_line, label_map) is not None:
                break  # ran into a different inline label
            add(field, next_line)
            break

    # Last-resort numeric fallbacks for the strongest patterns.
    if "nin" in schema.field_names and "nin" not in fields:
        m = re.search(r"\b(\d{11})\b", text)
        if m:
            add("nin", m.group(1))
    if "bvn" in schema.field_names and "bvn" not in fields:
        m = re.search(r"bvn[^0-9]{0,16}(\d{11})", text, re.IGNORECASE)
        if m:
            add("bvn", m.group(1))

    return FieldExtractionResult(fields=fields, schema_followed=False)

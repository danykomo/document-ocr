"""Field normalization + Nigerian validation rules.

Normalization makes scoring fair (a model that returns "ADA  OKAFOR" should not
be penalized against ground truth "Ada Okafor") and feeds the deterministic
consistency checks the spec calls for above raw model extraction.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

# Field "kinds" drive both normalization and scoring strategy.
NAME_FIELDS = {"full_name", "fullName", "first_name", "firstName", "middle_name",
               "middleName", "surname", "last_name"}
DATE_FIELDS = {"date_of_birth", "dateOfBirth", "issue_date", "issueDate",
               "expiry_date", "expiryDate", "dob"}
NUMBER_FIELDS = {"nin", "bvn", "passport_number", "passportNumber",
                 "drivers_license_number", "driversLicenseNumber",
                 "document_number", "documentNumber", "account_number",
                 "accountNumber"}


def field_kind(field: str) -> str:
    if field in NAME_FIELDS:
        return "name"
    if field in DATE_FIELDS:
        return "date"
    if field in NUMBER_FIELDS:
        return "number"
    if field in {"address"}:
        return "address"
    return "text"


_WS = re.compile(r"\s+")


def normalize_whitespace(value: str) -> str:
    return _WS.sub(" ", value).strip()


def normalize_name(value: str) -> str:
    cleaned = normalize_whitespace(value)
    # Collapse punctuation that OCR commonly inserts, keep hyphen/apostrophe.
    cleaned = re.sub(r"[^\w\s'\-]", "", cleaned)
    return cleaned.upper()


def normalize_number(value: str) -> str:
    # Identity numbers: keep digits only, but preserve a leading alpha prefix for
    # passport-style values (e.g. A01234567).
    stripped = value.strip().upper()
    alpha = re.match(r"^([A-Z]{0,2})", stripped)
    digits = re.sub(r"\D", "", stripped)
    prefix = alpha.group(1) if alpha else ""
    return f"{prefix}{digits}"


_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y", "%Y/%m/%d", "%d %b, %Y", "%d-%b-%Y", "%d/%m/%y",
]


def normalize_date(value: str) -> str:
    """Best-effort parse to ISO ``YYYY-MM-DD``; returns cleaned input on failure."""
    cleaned = normalize_whitespace(value)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    # Last resort: pull a YYYY-MM-DD or DD/MM/YYYY substring.
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", cleaned)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", cleaned)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
        except ValueError:
            pass
    return cleaned.upper()


def normalize_gender(value: str) -> str:
    v = value.strip().upper()
    if v in {"M", "MALE"}:
        return "MALE"
    if v in {"F", "FEMALE"}:
        return "FEMALE"
    return v


def normalize_field(field: str, value: Optional[str]) -> str:
    if value is None:
        return ""
    kind = field_kind(field)
    if field in {"gender", "sex"}:
        return normalize_gender(value)
    if kind == "name":
        return normalize_name(value)
    if kind == "date":
        return normalize_date(value)
    if kind == "number":
        return normalize_number(value)
    return normalize_whitespace(value).upper()


# --------------------------------------------------------------------------- #
# Deterministic Nigerian validation checks (spec: Consistency And Validation)
# --------------------------------------------------------------------------- #
def validate_nin(value: str) -> tuple[bool, str]:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 11:
        return True, "ok"
    return False, f"NIN should be 11 digits, got {len(digits)}"


def validate_bvn(value: str) -> tuple[bool, str]:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 11:
        return True, "ok"
    return False, f"BVN should be 11 digits, got {len(digits)}"


def validate_passport_number(value: str) -> tuple[bool, str]:
    v = (value or "").strip().upper()
    if re.fullmatch(r"[A-Z]\d{8}", v):
        return True, "ok"
    return False, "Passport number should be 1 letter + 8 digits"


def validate_expiry_not_past(value: str, today: Optional[date] = None) -> tuple[bool, str]:
    today = today or date.today()
    iso = normalize_date(value)
    try:
        d = date.fromisoformat(iso)
    except ValueError:
        return False, "Expiry date unparseable"
    if d < today:
        return False, f"Document expired on {iso}"
    return True, "ok"

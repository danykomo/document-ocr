"""Cover the parser fix that made GLM-OCR score 0 on a perfect read.

The model returns label-style JSON keys ("Surname", "Date of Birth", "Sex",
"Date of Issue"); the schema uses snake_case. The coercion normalizes both,
and synthesizes ``full_name`` from parts when the document doesn't show a
combined Name line.
"""

from document_ocr_benchmarks.models import DocumentType
from document_ocr_benchmarks.providers.vlm_openai import (
    _coerce_keys,
    _normalize_key,
)
from document_ocr_benchmarks.schemas import schema_for


def test_normalize_key_strips_punct_and_lowercases():
    assert _normalize_key("First Name") == "first_name"
    assert _normalize_key("Date of Birth") == "date_of_birth"
    assert _normalize_key("Passport No.") == "passport_no"


def test_coerce_keys_maps_glm_ocr_response_to_schema():
    """The exact response GLM-OCR returned on the user's NIN slip."""
    schema = schema_for(DocumentType.NIN_SLIP)
    parsed = {
        "Surname": "Onyeka",
        "First Name": "Halima",
        "Date of Birth": "1973-03-10",
        "Sex": "Male",
        "NIN": "26596939007",
        "Address": "23 Ahmadu Bello Way, GRA, Port Harcourt",
        "Date of Issue": "2022-04-29",
    }
    out = _coerce_keys(parsed, schema.field_names)
    assert out["surname"] == "Onyeka"
    assert out["first_name"] == "Halima"
    assert out["date_of_birth"] == "1973-03-10"
    assert out["gender"] == "Male"             # "Sex" alias
    assert out["nin"] == "26596939007"
    assert out["address"] == "23 Ahmadu Bello Way, GRA, Port Harcourt"
    assert out["issue_date"] == "2022-04-29"   # "Date of Issue" alias
    # full_name is not in the NIN_SLIP schema anymore (real slips show
    # surname/first/middle separately), so the coercer must not synthesise one.
    assert "full_name" not in out


def test_coerce_keys_does_not_override_explicit_full_name():
    schema = schema_for(DocumentType.UTILITY_BILL)
    parsed = {"Name": "Halima Oluwaseun Onyeka",
              "Address": "1 Test Street", "Issued By": "Ikeja Electric"}
    out = _coerce_keys(parsed, schema.field_names)
    assert out["full_name"] == "Halima Oluwaseun Onyeka"
    assert out["address"] == "1 Test Street"
    assert out["issuing_authority"] == "Ikeja Electric"


def test_coerce_keys_handles_object_value_form():
    """Object form ({"value":..,"confidence":..}) survives coercion intact."""
    schema = schema_for(DocumentType.NIN_SLIP)
    parsed = {
        "Surname": {"value": "Onyeka", "confidence": 0.9},
        "First Name": {"value": "Halima", "confidence": 0.9},
    }
    out = _coerce_keys(parsed, schema.field_names)
    assert out["surname"] == {"value": "Onyeka", "confidence": 0.9}
    assert out["first_name"] == {"value": "Halima", "confidence": 0.9}



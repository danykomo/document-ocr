"""Regression tests for the shared OCR-text → fields parser.

The big sample is PaddleOCR-VL's exact response on the user's real NIN slip —
proves the parser turns that into useful schema fields instead of 0.0.
"""

from document_ocr_benchmarks.models import DocumentType
from document_ocr_benchmarks.providers.text_kie import (
    _build_label_map,
    _norm_label,
    extract_fields_from_text,
)
from document_ocr_benchmarks.schemas import schema_for


def test_norm_label_collapses_punct():
    assert _norm_label("First Name") == "first name"
    assert _norm_label("Date of Birth:") == "date of birth"
    assert _norm_label("Sex.") == "sex"


def test_paddleocr_vl_real_output_parses_to_fields():
    """The exact text PaddleOCR-VL returned on a real NIN slip."""
    text = (
        "National Identity Management System\n"
        "Federal Republic of Nigeria\n"
        "National Identification Number Slip (NINS)\n"
        "Tracking ID:\n"
        "S7YoOoZSH0007A2\n"
        "Surname:\n"
        "OBENDE\n"
        "Address:\n"
        "NIN:\n"
        "93473966441\n"
        "First Name:\n"
        "ABOSEDE\n"
        "6 ODUNLAM! STREET\n"
        "Kosofe\n"
        "Middle Name: CHRISTIANA\n"
        "Gender:\n"
        "F\n"
        "Lagos\n"
    )
    schema = schema_for(DocumentType.NIN_SLIP)
    result = extract_fields_from_text(text, schema, source="paddleocr-vl")
    got = {k: v.value for k, v in result.fields.items()}

    # The five we should reliably get out of this messy column-interleaved layout:
    assert got["surname"] == "OBENDE"
    assert got["first_name"] == "ABOSEDE"
    assert got["middle_name"] == "CHRISTIANA"
    assert got["gender"] == "F"
    assert got["nin"] == "93473966441"
    # Address won't extract cleanly because the model interleaved label and
    # value across the layout columns — that's an honest limitation, not a bug.


def test_inline_and_standalone_label_forms_both_work():
    schema = schema_for(DocumentType.NIN_SLIP)
    text = "Surname: OKAFOR\nFirst Name:\nADA\nNIN: 12345678901\n"
    got = {k: v.value for k, v in extract_fields_from_text(text, schema, "x").fields.items()}
    assert got["surname"] == "OKAFOR"
    assert got["first_name"] == "ADA"
    assert got["nin"] == "12345678901"


def test_nin_fallback_regex_when_label_missing():
    schema = schema_for(DocumentType.NIN_SLIP)
    text = "Some preamble. 93473966441 appears here without a label."
    got = {k: v.value for k, v in extract_fields_from_text(text, schema, "x").fields.items()}
    assert got["nin"] == "93473966441"


def test_label_map_is_scoped_to_schema():
    """Bank-mandate-only labels (bvn) shouldn't be in an NIN-slip schema map."""
    nin = _build_label_map(schema_for(DocumentType.NIN_SLIP))
    mandate = _build_label_map(schema_for(DocumentType.BANK_MANDATE))
    assert "bvn" not in nin and "bvn" in mandate

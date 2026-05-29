from document_ocr_benchmarks.models import (
    ExpectedFields,
    FieldExtractionResult,
    FieldValue,
    DocumentType,
    ProviderRunResult,
)
from document_ocr_benchmarks.providers.mock import MockRunner
from document_ocr_benchmarks.schemas import schema_for
from document_ocr_benchmarks.scoring import score_field, score_sample


def test_score_field_exact_and_fuzzy():
    assert score_field("nin", "12345678901", "123 456 789 01") == 1.0
    assert score_field("nin", "12345678901", "12345678900") == 0.0  # numbers strict
    assert score_field("full_name", "Ada Okafor", "Ada Okafor") == 1.0
    partial = score_field("full_name", "Ada Okafor", "Ada Okafer")
    assert 0.5 < partial < 1.0  # one-char slip -> partial credit


def test_score_field_date_strict():
    assert score_field("date_of_birth", "1990-03-14", "14/03/1990") == 1.0
    assert score_field("date_of_birth", "1990-03-14", "1991-03-14") == 0.0


def _perfect_run(expected: ExpectedFields) -> ProviderRunResult:
    fields = {
        k: FieldValue(value=v, normalized_value=v, confidence=1.0, source="oracle")
        for k, v in expected.fields.items()
    }
    return ProviderRunResult(
        provider="oracle",
        fields=FieldExtractionResult(fields=fields, schema_followed=True),
        latency_ms={"total": 100.0},
    )


def test_score_sample_perfect_extraction():
    expected = ExpectedFields(
        sample_id="x",
        document_type=DocumentType.NIN_SLIP,
        fields={"full_name": "Ada Okafor", "nin": "12345678901",
                "date_of_birth": "1990-03-14"},
        required_fields=["full_name", "nin", "date_of_birth"],
    )
    run = _perfect_run(expected)
    result = score_sample(
        provider=MockRunner(), run=run, expected=expected,
        schema=schema_for(DocumentType.NIN_SLIP),
    )
    assert result.field_accuracy == 1.0
    assert result.required_field_recall == 1.0
    assert result.schema_followed is True

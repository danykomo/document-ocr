"""Scoring: turn provider output + ground truth into a BenchmarkResult.

Scoring is normalization-aware so a provider isn't penalized for casing/spacing
or date formatting. Numbers and dates are scored strictly (exact after
normalization); names/addresses/text get partial credit via fuzzy similarity,
reflecting that a one-character OCR slip on a long name is not a total miss.
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz.fuzz import token_sort_ratio

from .models import (
    BenchmarkResult,
    CaptureCondition,
    DocumentType,
    ExpectedFields,
    ProviderRunResult,
)
from .normalization import field_kind, normalize_field
from .providers.base import ProviderRunner
from .schemas import DocumentSchema

# A required field counts as "recalled" at or above this score.
RECALL_THRESHOLD = 0.85


def score_field(field: str, expected: str, predicted: Optional[str]) -> float:
    """Return a match score in [0, 1] for one field."""
    exp_norm = normalize_field(field, expected)
    if not predicted:
        return 0.0
    pred_norm = normalize_field(field, predicted)
    if not exp_norm:
        return 1.0 if not pred_norm else 0.0
    if exp_norm == pred_norm:
        return 1.0
    kind = field_kind(field)
    if kind in {"number", "date"}:
        # Strict: a wrong NIN/passport/date is a miss, not partial credit.
        return 0.0
    # Names / addresses / free text: fuzzy partial credit.
    return round(token_sort_ratio(exp_norm, pred_norm) / 100.0, 3)


def score_sample(
    *,
    provider: ProviderRunner,
    run: ProviderRunResult,
    expected: ExpectedFields,
    schema: Optional[DocumentSchema],
    gpu_memory_mb: Optional[float] = None,
    peak_memory_mb: Optional[float] = None,
) -> BenchmarkResult:
    """Combine one provider run with ground truth into an evidence record."""
    per_field: dict[str, float] = {}
    extracted = run.fields.fields if run.fields else {}

    score_fields = list(expected.fields.keys())
    for fname in score_fields:
        pred = extracted.get(fname)
        pred_val = pred.value if pred else None
        per_field[fname] = score_field(fname, expected.fields[fname], pred_val)

    field_accuracy = (
        round(sum(per_field.values()) / len(per_field), 4) if per_field else 0.0
    )

    required = expected.required_fields or (schema.required_field_names if schema else [])
    if required:
        recalled = sum(1 for f in required if per_field.get(f, 0.0) >= RECALL_THRESHOLD)
        required_recall = round(recalled / len(required), 4)
    else:
        required_recall = 0.0

    # Classification.
    classification_correct: Optional[bool] = None
    classification_conf = 0.0
    if run.classification is not None:
        classification_correct = run.classification.document_type == expected.document_type
        classification_conf = run.classification.confidence

    # Portrait.
    portrait_extracted = bool(run.portrait and run.portrait.available)
    portrait_q = run.portrait.quality_score if run.portrait else 0.0

    notes: list[str] = []
    if run.skipped_capabilities:
        notes.append(f"skipped: {', '.join(run.skipped_capabilities)}")
    if expected.portrait_present and not portrait_extracted:
        notes.append("expected portrait not extracted")

    return BenchmarkResult(
        candidate=provider.name,
        document_type=expected.document_type,
        sample_id=expected.sample_id,
        classification_correct=classification_correct,
        classification_confidence=classification_conf,
        field_accuracy=field_accuracy,
        required_field_recall=required_recall,
        per_field_scores=per_field,
        schema_followed=bool(run.fields and run.fields.schema_followed),
        portrait_expected=expected.portrait_present,
        portrait_extracted=portrait_extracted,
        portrait_quality_score=portrait_q,
        text_char_count=len(run.text.raw) if run.text else 0,
        latency_ms=run.latency_ms.get("total", 0.0),
        latency_breakdown=run.latency_ms,
        peak_memory_mb=peak_memory_mb,
        gpu_memory_mb=gpu_memory_mb,
        cost_estimate=provider.cost_estimate,
        license_status=provider.license_status,
        on_prem=provider.on_prem,
        error=run.error,
        error_code=run.error_code,
        provider_version=run.provider_version,
        model_version=run.model_version,
        notes=notes,
    )

"""Provider-neutral data models shared across the harness.

These mirror the contracts described in the service spec (analyze request /
response, field values with provenance, the benchmark evidence record) so the
benchmark harness and the eventual ``document-ocr-core`` package speak the same
language.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class DocumentType(str, Enum):
    """Normalized document types (spec: Document Types)."""

    NATIONAL_ID = "national_id"
    NIN_SLIP = "nin_slip"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    VOTER_ID = "voter_id"
    BANK_MANDATE = "bank_mandate"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    CAC_DOCUMENT = "cac_document"
    OTHER = "other"
    UNKNOWN = "unknown"


class CaptureCondition(str, Enum):
    """How a sample was captured. Drives the benchmark robustness groups."""

    CLEAN = "clean"
    MOBILE = "mobile"
    LOW_LIGHT = "low_light"
    GLARE = "glare"
    CROPPED = "cropped"
    ROTATED = "rotated"
    LAMINATED = "laminated"
    PHOTOCOPY = "photocopy"
    WHATSAPP = "whatsapp"
    BLURRED = "blurred"


class Capability(str, Enum):
    """Canonical provider operations (spec: Provider Interface)."""

    CLASSIFY = "classify"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_FIELDS = "extract_fields"
    EXTRACT_PORTRAIT = "extract_portrait"
    ASSESS_QUALITY = "assess_quality"


class LicenseStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class CostEstimate(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --------------------------------------------------------------------------- #
# Provider result models
# --------------------------------------------------------------------------- #
class FieldValue(BaseModel):
    """A single extracted field with provenance (spec: Nigerian Field Intelligence)."""

    value: Optional[str] = None
    normalized_value: Optional[str] = None
    confidence: float = 0.0
    source: Optional[str] = None
    bbox: Optional[list[int]] = None
    validation_status: Optional[str] = None


class ClassificationResult(BaseModel):
    document_type: DocumentType = DocumentType.UNKNOWN
    label: Optional[str] = None
    confidence: float = 0.0


class TextBlock(BaseModel):
    text: str
    confidence: float = 0.0
    bbox: Optional[list[int]] = None


class TextExtractionResult(BaseModel):
    raw: str = ""
    blocks: list[TextBlock] = Field(default_factory=list)


class FieldExtractionResult(BaseModel):
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    # Whether the provider returned a parseable response matching the requested
    # schema. A core decision dimension: deterministic JSON/schema-following.
    schema_followed: bool = False


class PortraitResult(BaseModel):
    available: bool = False
    bbox: Optional[list[int]] = None
    confidence: float = 0.0
    source: Optional[str] = None
    quality_score: float = 0.0
    # Base64 is deliberately optional and dropped from persisted evidence.
    image_base64: Optional[str] = None


class QualityResult(BaseModel):
    readable: bool = True
    blur: str = "unknown"        # low | medium | high
    glare: str = "unknown"       # low | medium | high
    cropped: bool = False
    orientation: str = "unknown"  # upright | rotated | unknown
    quality_score: float = 0.0


class ProviderRunResult(BaseModel):
    """Everything one provider produced for one sample, with per-op timings."""

    provider: str
    provider_version: Optional[str] = None
    model_version: Optional[str] = None

    classification: Optional[ClassificationResult] = None
    text: Optional[TextExtractionResult] = None
    fields: Optional[FieldExtractionResult] = None
    portrait: Optional[PortraitResult] = None
    quality: Optional[QualityResult] = None

    latency_ms: dict[str, float] = Field(default_factory=dict)  # per-op + total
    fallbacks_used: list[str] = Field(default_factory=list)
    skipped_capabilities: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    error_code: Optional[str] = None


class CapabilityReport(BaseModel):
    provider: str
    available: bool
    reason: Optional[str] = None
    capabilities: list[Capability] = Field(default_factory=list)
    license_name: Optional[str] = None
    license_status: LicenseStatus = LicenseStatus.NEEDS_REVIEW
    cost_estimate: CostEstimate = CostEstimate.MEDIUM
    on_prem: bool = True
    deployment_notes: Optional[str] = None


# --------------------------------------------------------------------------- #
# Dataset models (manifest / expected ground truth)
# --------------------------------------------------------------------------- #
class Sample(BaseModel):
    sample_id: str
    document_type: DocumentType
    image_path: str  # relative to the dataset root
    side: str = "front"
    capture_condition: CaptureCondition = CaptureCondition.CLEAN
    source: str = "synthetic"  # synthetic | real
    notes: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class ExpectedFields(BaseModel):
    """Ground truth for one sample."""

    sample_id: str
    document_type: DocumentType
    fields: dict[str, str] = Field(default_factory=dict)
    required_fields: list[str] = Field(default_factory=list)
    portrait_present: bool = False
    portrait_bbox: Optional[list[int]] = None


class Manifest(BaseModel):
    name: str = "document-ocr-benchmark"
    version: str = "v1"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: Optional[str] = None
    samples: list[Sample] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Benchmark evidence record (spec: QA And Benchmarking)
# --------------------------------------------------------------------------- #
class BenchmarkResult(BaseModel):
    """Structured, repeatable evidence for one (candidate, sample) run.

    Serialized with camelCase aliases to match the spec's evidence JSON shape.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    candidate: str
    document_type: DocumentType
    sample_id: str
    capture_condition: CaptureCondition = CaptureCondition.CLEAN

    # Accuracy
    classification_correct: Optional[bool] = None
    classification_confidence: float = 0.0
    field_accuracy: float = 0.0
    required_field_recall: float = 0.0
    per_field_scores: dict[str, float] = Field(default_factory=dict)
    schema_followed: bool = False

    # Portrait
    portrait_expected: bool = False
    portrait_extracted: bool = False
    portrait_quality_score: float = 0.0

    # Text
    text_char_count: int = 0

    # Resources
    latency_ms: float = 0.0
    latency_breakdown: dict[str, float] = Field(default_factory=dict)
    peak_memory_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None

    # Commercial / operational posture (carried from provider metadata)
    cost_estimate: CostEstimate = CostEstimate.MEDIUM
    license_status: LicenseStatus = LicenseStatus.NEEDS_REVIEW
    on_prem: bool = True

    # Diagnostics (redaction-safe: no raw images, no raw prompts)
    error: Optional[str] = None
    error_code: Optional[str] = None
    provider_version: Optional[str] = None
    model_version: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

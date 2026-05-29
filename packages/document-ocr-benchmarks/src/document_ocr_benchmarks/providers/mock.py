"""Deterministic mock provider.

Has no dependencies and always available. Used to smoke-test the orchestration,
scoring, and reporting pipeline without any OCR backend. It does not read ground
truth, so scores against real samples will be low by design.
"""

from __future__ import annotations

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
    TextExtractionResult,
)
from ..schemas import DocumentSchema
from .base import ProviderRunner


class MockRunner(ProviderRunner):
    name = "mock"
    display_name = "Mock provider (pipeline test)"
    license_name = "n/a"
    license_status = LicenseStatus.APPROVED
    cost_estimate = CostEstimate.LOW
    on_prem = True
    deployment_notes = "Test-only stub. Returns canned data; not a real OCR engine."

    def capabilities(self) -> set[Capability]:
        return {
            Capability.CLASSIFY,
            Capability.EXTRACT_TEXT,
            Capability.EXTRACT_FIELDS,
            Capability.EXTRACT_PORTRAIT,
            Capability.ASSESS_QUALITY,
        }

    def classify(self, image: LoadedImage) -> ClassificationResult:
        return ClassificationResult(
            document_type=DocumentType.NIN_SLIP, label="nin_slip", confidence=0.5
        )

    def extract_text(self, image: LoadedImage) -> TextExtractionResult:
        return TextExtractionResult(raw="MOCK OCR TEXT")

    def extract_fields(
        self, image: LoadedImage, schema: DocumentSchema
    ) -> FieldExtractionResult:
        fields = {
            schema.field_names[0]: FieldValue(
                value="MOCK VALUE",
                normalized_value="MOCK VALUE",
                confidence=0.5,
                source=self.name,
            )
        }
        return FieldExtractionResult(fields=fields, schema_followed=True)

    def extract_portrait(self, image: LoadedImage) -> PortraitResult:
        return PortraitResult(available=False, source=self.name)

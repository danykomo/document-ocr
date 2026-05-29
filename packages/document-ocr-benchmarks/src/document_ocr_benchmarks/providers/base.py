"""Provider runner interface.

Every OCR/VLM candidate is wrapped in a ``ProviderRunner``. Providers may
implement only part of the canonical operation set; unsupported operations are
reported explicitly (spec: "The service must report unsupported capabilities
clearly rather than silently returning empty results.").
"""

from __future__ import annotations

import time
from typing import Optional

from ..imaging import LoadedImage, assess_quality
from ..models import (
    Capability,
    CapabilityReport,
    ClassificationResult,
    CostEstimate,
    FieldExtractionResult,
    LicenseStatus,
    PortraitResult,
    ProviderRunResult,
    QualityResult,
    TextExtractionResult,
)
from ..schemas import DocumentSchema


class UnsupportedCapability(NotImplementedError):
    """Raised by a provider when asked to perform an op it does not support."""


class ProviderRunner:
    """Base class for all provider adapters.

    Subclasses set the class-level metadata and override the operations they
    support. Metadata is used to populate benchmark evidence (license, cost,
    on-prem posture) even when the provider is not configured to run.
    """

    name: str = "base"
    display_name: str = "Base Provider"
    license_name: str = "unknown"
    license_status: LicenseStatus = LicenseStatus.NEEDS_REVIEW
    cost_estimate: CostEstimate = CostEstimate.MEDIUM
    on_prem: bool = True
    deployment_notes: str = ""
    provider_version: Optional[str] = None
    model_version: Optional[str] = None

    # -- capability + availability ---------------------------------------- #
    def capabilities(self) -> set[Capability]:
        return set()

    def is_available(self) -> tuple[bool, Optional[str]]:
        """Return (available, reason). Default: available."""
        return True, None

    def capability_report(self) -> CapabilityReport:
        ok, reason = self.is_available()
        return CapabilityReport(
            provider=self.name,
            available=ok,
            reason=reason,
            capabilities=sorted(self.capabilities(), key=lambda c: c.value),
            license_name=self.license_name,
            license_status=self.license_status,
            cost_estimate=self.cost_estimate,
            on_prem=self.on_prem,
            deployment_notes=self.deployment_notes,
        )

    # -- canonical operations (override what you support) ----------------- #
    def classify(self, image: LoadedImage) -> ClassificationResult:
        raise UnsupportedCapability(f"{self.name} does not support classify")

    def extract_text(self, image: LoadedImage) -> TextExtractionResult:
        raise UnsupportedCapability(f"{self.name} does not support extract_text")

    def extract_fields(
        self, image: LoadedImage, schema: DocumentSchema
    ) -> FieldExtractionResult:
        raise UnsupportedCapability(f"{self.name} does not support extract_fields")

    def extract_portrait(self, image: LoadedImage) -> PortraitResult:
        raise UnsupportedCapability(f"{self.name} does not support extract_portrait")

    def assess_quality(self, image: LoadedImage) -> QualityResult:
        # Default to the shared deterministic assessor. Providers rarely need
        # to override this.
        return assess_quality(image)

    # -- orchestration ---------------------------------------------------- #
    def analyze(
        self,
        image: LoadedImage,
        schema: Optional[DocumentSchema],
        *,
        do_classify: bool = True,
        do_text: bool = True,
        do_fields: bool = True,
        do_portrait: bool = True,
        do_quality: bool = True,
    ) -> ProviderRunResult:
        """Run all requested ops, timing each, degrading gracefully.

        Each op is isolated: a failure or unsupported op is recorded and the
        rest still run. The total reflects wall-clock across attempted ops.
        """
        result = ProviderRunResult(
            provider=self.name,
            provider_version=self.provider_version,
            model_version=self.model_version,
        )
        caps = self.capabilities()
        total_start = time.perf_counter()
        errors: list[str] = []

        def _run(cap: Capability, fn):
            if cap not in caps:
                result.skipped_capabilities.append(cap.value)
                return None
            start = time.perf_counter()
            try:
                return fn()
            except UnsupportedCapability:
                result.skipped_capabilities.append(cap.value)
                return None
            except Exception as exc:  # normalized; raw provider error never leaks
                errors.append(f"{cap.value}: {type(exc).__name__}: {exc}")
                return None
            finally:
                result.latency_ms[cap.value] = round(
                    (time.perf_counter() - start) * 1000, 1
                )

        # Quality is always available (shared deterministic assessor).
        if do_quality:
            q_start = time.perf_counter()
            try:
                result.quality = self.assess_quality(image)
            except Exception as exc:
                errors.append(f"assess_quality: {type(exc).__name__}: {exc}")
            result.latency_ms["assess_quality"] = round(
                (time.perf_counter() - q_start) * 1000, 1
            )
        if do_classify:
            result.classification = _run(Capability.CLASSIFY, lambda: self.classify(image))
        if do_text:
            result.text = _run(Capability.EXTRACT_TEXT, lambda: self.extract_text(image))
        if do_fields and schema is not None:
            result.fields = _run(
                Capability.EXTRACT_FIELDS, lambda: self.extract_fields(image, schema)
            )
        if do_portrait:
            result.portrait = _run(
                Capability.EXTRACT_PORTRAIT, lambda: self.extract_portrait(image)
            )

        if errors:
            # Keep a normalized, redaction-safe summary; do not surface to clients.
            result.error = "; ".join(errors)
            result.error_code = "PROVIDER_PARTIAL_FAILURE"
        result.latency_ms["total"] = round((time.perf_counter() - total_start) * 1000, 1)
        return result

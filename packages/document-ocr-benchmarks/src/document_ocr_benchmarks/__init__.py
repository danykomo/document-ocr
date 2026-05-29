"""Document OCR benchmark harness.

Provider-neutral tooling to evaluate VLM/OCR candidates (GLM-OCR, Qwen-VL,
PaddleOCR-VL, olmOCR, classical fallback) against Nigerian banking and identity
documents. The harness measures more than raw accuracy: it captures latency,
resource use, schema-following reliability, portrait extraction, robustness to
poor captures, and records license/deployment posture as structured evidence.

See ``DOCUMENT_OCR_ENGINE_SERVICE_SPEC.md`` for the product context.
"""

__version__ = "0.1.0"

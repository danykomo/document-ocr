# Document OCR Engine Service Specification

> This is the canonical product/architecture specification for the Document OCR
> Engine. The current repository implements the **benchmark spike** mandated by
> the "Benchmark-First Delivery Rule" below. See the root `README.md` for what is
> built so far and `docs/` for benchmark + deployment guides.

## Purpose

This document defines the target architecture for a standalone Document OCR Engine
that can be used independently by multiple Innovantics products and also plugged
into Bifense identity-verification workflows.

The OCR capability must not be designed as a Bifense-only feature. It should be a
product-grade, self-contained engine with its own API contract, deployment model,
provider abstraction, and operational boundary.

Primary consumers:

- Bifense, for document-backed identity verification and face match flows
- Banklet, for customer onboarding, account opening, and document data capture
- Banks or enterprise customers that need only OCR without the rest of Bifense
- Future Innovantics products that need document classification, text extraction,
  field extraction, or document portrait extraction

## Product Differentiation Strategy

The differentiator is **Nigeria-native document intelligence** for banking,
fintech, and identity workflows — not another generic OCR API.

Generic OCR answers: *what text is visible in this image?*

The Innovantics OCR engine should answer:

- what Nigerian document is this?
- what identity or banking fields can be trusted?
- which fields are missing, low-confidence, or conflicting?
- is the document good enough for onboarding?
- is there a document portrait that Bifense can compare against a selfie or enrolled face?
- does this need authority verification, manual review, rejection, or recapture?

First commercial wedge (narrow and excellent): **NIN slip, passport, driver's
licence, and bank mandate OCR for Nigerian onboarding.**

### Nigeria Document Pack

NIN slip, improved NIN slip, Nigerian national ID, Nigerian passport data page,
Nigerian driver's licence, voter's card, BVN/NIN onboarding forms, bank mandate
forms, utility bills, bank statements, CAC documents.

Each supported document should have: normalized document type, field schema, field
validation rules, expected layout hints, portrait extraction rules where
applicable, quality requirements, sample benchmark set, known failure modes.

### Nigerian Field Intelligence

Extract structured fields, not just raw text. Priority fields: full name, first
name, middle name, surname, date of birth, gender, NIN, BVN, passport number,
driver's licence number, document number, issue date, expiry date, address,
issuing authority, document portrait.

Every extracted field should include: raw value, normalized value, confidence,
source provider, bounding box when available, validation status, review
recommendation when low confidence.

### Nigerian Consistency And Validation Rules

Deterministic checks above model extraction: NIN/BVN length and numeric format,
passport/driver's-licence format sanity, expiry presence and not-expired (where
policy requires), plausible DOB, name/address similarity across documents and
forms, document type matches the workflow, required side present, portrait
present/usable where face match is required. Output review-friendly signals
(status + coded checks with severity, message, confidence).

### Authority-Aware But Not Authority-Dependent

OCR owns: field extraction, normalization, confidence, document quality,
authority-routing hints. Product adapters own: NIBSS/BVN, NIMC/NIN, FRSC, passport
authority verification, customer credentials, final decision policy. This keeps
the OCR engine useful even where authority integration is unavailable.

### Portrait Extraction As A Bifense Advantage

Document portrait extraction is a first-class capability so Bifense can compose
portrait-vs-selfie, portrait-vs-enrolled-image, portrait-vs-watchlist, quality
scoring, and liveness + document-face match — without forcing OCR to become a
biometric service.

### On-Prem And Data Residency Differentiator

Support self-hosted deployment, local-only model mode, no external OCR provider
calls, no source-image logging, configurable artifact retention, customer-managed
storage, audit logs with protected fields redacted, model/provider selection by
deployment policy.

### Nigerian Benchmark Suite

Provider choice must be based on Nigerian document performance, not public
leaderboards. The benchmark suite measures GLM-OCR, Qwen-VL, PaddleOCR-VL, olmOCR,
and fallback OCR against Nigerian banking and identity samples across clean scans,
mobile/low-light/glare captures, cropped/rotated/laminated docs, photocopies,
WhatsApp-style compressed images, and mixed casing/spacing. It produces repeatable
evidence: per-field accuracy, portrait quality, latency/hardware cost,
schema-following reliability, failure behavior, license suitability, on-prem
feasibility. This benchmark suite is part of the product moat.

## Core Design Decision

Build OCR as an independent service and engine:

- OCR owns document understanding.
- Bifense owns biometric identity verification.
- Banklet owns banking workflow and customer onboarding.
- Product-specific behavior lives in adapters around the OCR service, not inside
  the OCR engine.

The OCR engine exposes neutral document-processing contracts and must not require
Bifense tenant models, billing tables, biometric workers, or Banklet banking
tables to run.

## Product Boundary

**OCR Engine Responsibilities:** document classification, raw OCR text extraction,
structured field extraction, document portrait extraction, image quality
assessment, page/side detection, provider selection and fallback, confidence
scoring, normalized output, optional async job processing, provider diagnostics
and safe error normalization.

**OCR Engine Non-Responsibilities:** Bifense face comparison/liveness/IDV
decisions, tenant entitlements, wallet debit; Banklet account creation/approval;
final KYC approval/rejection; authority-backed BVN/NIN/passport/national-ID
verification; long-term system-of-record behavior.

**Product Adapter Responsibilities:** each consuming product (Bifense, Banklet,
standalone bank) owns a thin adapter mapping its context to OCR and applying its
own policy, retention, billing, and review.

## Architecture

MVP is Python-first (strongest OCR/VLM ecosystem; faster benchmarking; simpler GPU
serving/preprocessing/iteration). Bifense and Banklet consume OCR over HTTP +
generated clients, so the implementation language does not leak into product code.

Recommended runtime shape:

```
apps/ocr-service                 FastAPI HTTP service, auth, async jobs, health, OpenAPI
packages/document-ocr-core       provider-neutral models, interfaces, normalization, confidence
packages/document-ocr-client     generated TS/Go/Python clients
packages/document-ocr-benchmarks Python benchmark harness, manifests, runners, scoring  <-- THIS REPO
products/bifense adapter         orchestration, billing, audit, retention, biometric compare
products/banklet adapter         onboarding and account-opening integration
```

### Python Service Stack

Python 3.12+, FastAPI, Pydantic, Uvicorn/Gunicorn, OpenCV + Pillow, PyTorch +
Transformers, vLLM where supported, PaddleOCR + Tesseract for fallback, Redis +
RQ/Celery/Dramatiq/Arq for async, OpenTelemetry-compatible logging/metrics,
OpenAPI-generated clients. Node should not own OCR inference; Go can be a thin
gateway later.

### Benchmark-First Delivery Rule

> Deep service development should not start before the VLM/OCR candidates are
> evaluated. **The first implementation milestone is a benchmark spike, not a
> complete OCR service.**

Benchmark spike deliverables: runnable Python environment, sample manifest format,
provider runner interface, at least two provider runners, scoring script, JSON
result output, short recommendation report, license/deployment notes per candidate.

## Deployment Modes

1. **Standalone OCR Service** — bank/product needs OCR only; own OpenAPI; on-prem or cloud.
2. **Bifense-Plugged** — deployed beside Bifense; called via `DOCUMENT_OCR_SERVICE_URL`.
3. **Banklet-Plugged** — called via typed client; OCR unaware of Banklet models.
4. **Embedded Library Mode** — future optimization; service-first API stays canonical.

## OCR Provider Model

Multiple providers behind one interface. Initial candidates: GLM-OCR, Qwen-VL /
Qwen3-VL, PaddleOCR-VL, olmOCR, classical fallback (Tesseract/PaddleOCR), cloud
providers where policy allows, future fine-tuned Nigerian model. No model is
hard-coded; the default is chosen from measured results + license/deployment/cost.

Canonical provider operations: `AnalyzeDocument`, `ClassifyDocument`, `ExtractText`,
`ExtractFields`, `ExtractPortrait`, `AssessQuality`. Providers may support only part
of the interface; the service must report unsupported capabilities clearly rather
than silently returning empty results.

Provider selection supports default-per-deployment, override by tenant/customer
policy, override by document type, local-only mode, and fallback chains with
preserved provenance. The benchmark winner may vary by operation; the architecture
should allow different defaults for `extract_text`, `extract_fields`, `classify`,
and `extract_portrait`.

## Document Types

`national_id`, `nin_slip`, `passport`, `drivers_license`, `voter_id`,
`bank_mandate`, `utility_bill`, `bank_statement`, `cac_document`, `other`,
`unknown`. Customer-specific aliases allowed; normalized types stored in responses.

## Core API Surface

Versioned base path `/api/v1`. Endpoints: `GET /health`, `/health/live`,
`/health/ready`; `POST /api/v1/documents/{analyze,classify,extract-text,
extract-fields,extract-portrait,quality}`; `POST /api/v1/jobs`,
`GET /api/v1/jobs/{jobId}`. MVP can implement `analyze` first and wrap the narrower
endpoints around the same orchestration path.

Canonical analyze request/response carry: document (+ hints), options, fieldSchema,
callerContext; and responses with document classification, per-field value +
normalizedValue + confidence + source + bbox, text blocks, optional portrait,
quality, engine/provider provenance, and processing time. Every extracted field
must include confidence; provider-derived fields include source; portrait is
explicitly marked unavailable when not found; debug output excludes raw prompts.

## Async Job Model

Statuses: `queued`, `processing`, `completed`, `failed`, `cancelled`, `expired`.
Support async from the beginning even if MVP mostly uses sync.

## Error Model

Client-safe canonical error shape with codes: `INVALID_IMAGE`,
`UNSUPPORTED_CONTENT_TYPE`, `DOCUMENT_UNREADABLE`, `DOCUMENT_TYPE_UNSUPPORTED`,
`DOCUMENT_TYPE_UNKNOWN`, `FIELD_EXTRACTION_FAILED`, `PORTRAIT_NOT_FOUND`,
`QUALITY_TOO_LOW`, `PROVIDER_UNAVAILABLE`, `PROVIDER_TIMEOUT`, `POLICY_REJECTED`,
`PAYLOAD_TOO_LARGE`, `RATE_LIMITED`, `INTERNAL_ERROR`. Provider raw errors are
logged internally and normalized before returning to clients.

## Security And Privacy

Three postures: internal service-to-service, standalone authenticated customer
API, on-prem private network. Controls: API key/token auth, request size limits,
content-type validation, file sanity checks, caller isolation, structured audit
logging, no raw document logging, no raw provider prompts in normal logs,
configurable retention, local-only provider mode, optional customer-managed
storage. NIN, BVN, passport number, DOB are protected identity data everywhere.

## Retention And Artifact Storage

Two modes: **stateless extraction** (MVP default — discard source/derived after
the request) and **managed artifact storage** (optional, policy-driven by the
consuming product/deployment). OCR does not assume Bifense retention defaults.

## Billing And Metering Boundary

OCR emits usage events (`ocr.document.analyzed` with operations, provider, timing);
product billing maps events to prices. *OCR records what happened; the consuming
product decides what it costs.*

## Integrations

**Bifense IDV flow** and **Banklet onboarding flow** call OCR for classification,
fields, quality, and optional portrait; each composes its own final decision
outside OCR. Standalone bank OCR exposes API-key access, customer retention policy,
OpenAPI/Postman, usage reporting, on-prem guide — with no Bifense/biometric
dependencies.

## Observability

Metrics: request counts by endpoint/product, document-type distribution, provider
latency/error rate, extraction + portrait success rates, low-quality rate, async
queue depth/duration. Logs exclude raw images and unredacted protected fields.

## Configuration

`OCR_SERVICE_PORT`, `OCR_DEFAULT_PROVIDER`, `OCR_ALLOWED_PROVIDERS`,
`OCR_LOCAL_ONLY_MODE`, `OCR_MAX_IMAGE_BYTES`, `OCR_SYNC_TIMEOUT_MS`,
`OCR_ASYNC_ENABLED`, `OCR_ARTIFACT_STORAGE_ENABLED`, `OCR_ARTIFACT_RETENTION_DAYS`,
`OCR_DEBUG_OUTPUT_ENABLED`, provider endpoints/keys, `OCR_PROVIDER_TESSERACT_ENABLED`,
`OCR_PROVIDER_PADDLEOCR_ENABLED`. Adapters add `DOCUMENT_OCR_SERVICE_URL`,
`DOCUMENT_OCR_SERVICE_API_KEY`, `DOCUMENT_OCR_REQUEST_TIMEOUT_MS`.

## QA And Benchmarking

Benchmark dimensions: raw OCR quality, key-field extraction quality, layout
understanding, portrait quality, classification accuracy, latency, memory/GPU,
hosting cost, on-prem feasibility, failure behavior on poor images, privacy/
data-residency posture, commercial license suitability, deterministic JSON/schema
following, setup complexity, CPU/GPU serving profile, batch throughput, confidence
calibration, redaction-safe diagnostics.

Benchmark evidence record (per candidate × sample): candidate, documentType,
sampleId, fieldAccuracy, requiredFieldRecall, portraitExtracted,
portraitQualityScore, latencyMs, gpuMemoryMb, costEstimate,
licenseStatus(approved|needs_review|rejected), notes.

Selection rule: pick a default only after testing on our target documents; reject
any candidate with unacceptable license/data-residency posture even if its score is
high; prefer a smaller local model when accuracy is close enough; allow a
two-provider strategy if one model is better for field extraction and another for
portrait extraction.

## Implementation Order

1. Python benchmark harness for GLM-OCR, Qwen-VL, PaddleOCR-VL, olmOCR, fallback. **(this repo)**
2. First Nigerian document sample manifest + expected-field schema. **(this repo)**
3. Run the benchmark on available samples. **(this repo)**
4. Record accuracy, latency, resource use, license, deployment constraints. **(this repo)**
5. Select the initial provider strategy from evidence.
6. `document-ocr-core` models + provider interfaces around the selected strategy.
7. FastAPI OCR service with health endpoints and `POST /api/v1/documents/analyze`.
8. Normalized field schemas for the first document types.
9. Portrait extraction.
10. Async job support if benchmark latency requires it.
11. Typed client package.
12. Bifense adapter + IDV orchestration.
13. Banklet adapter.
14. Standalone deployment docs, OpenAPI, Postman.

## Acceptance Criteria

OCR runs without Bifense; deployable as its own service; stable provider-neutral
API; Bifense and Banklet call it via client/adapter; responses include confidence
and provenance; provider raw errors not exposed; no biometric worker licenses; no
Bifense billing/tenant assumptions; standalone deployments have their own
OpenAPI/Postman; provider choice stays configurable and benchmark-driven.

## Open Questions For Implementation Planning

1. Which two providers should be wired into the first benchmark spike?
2. What real document samples can be legally used for benchmarking?
3. What minimum benchmark score is acceptable for NIN slip, passport, driver's
   licence, and bank mandate extraction?
4. Should artifact storage be included in MVP or deferred until Bifense IDV?
5. Which product consumes OCR first: Bifense, Banklet, or standalone bank OCR?

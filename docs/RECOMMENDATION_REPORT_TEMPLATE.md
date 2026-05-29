# Provider Recommendation — <DATE>

> The harness auto-generates `report.md` + `summary.json` per run. This template
> is the **human decision record** that wraps that evidence into a committed
> provider decision. Fill it in after running the benchmark on the agreed sample
> set (synthetic + real where legally available).

## Decision

- **Default provider:** `<provider>`
- **Per-operation strategy:**
  - `classify` → `<provider>`
  - `extract_text` → `<provider>`
  - `extract_fields` → `<provider>`
  - `extract_portrait` → `<provider>` (face detector if delegated)
- **Fallback chain:** `<e.g. glm-ocr -> qwen-vl -> tesseract>`
- **Status:** approved / provisional / needs more samples

## Evidence

- Benchmark run id: `<run_id>`
- Dataset: `<manifest name>` — `<N>` samples (`<synthetic/real split>`)
- Hardware: `<GPU model, VRAM>`
- Report: `benchmarks/document-ocr/results/<run_id>/report.md`

| Candidate | Composite | Field acc | Req. recall | Classify | Portrait | p50 ms | GPU MB | Cost | License |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| ... | | | | | | | | | |

## Per-document-type acceptance check

Set the minimum acceptable field accuracy per priority document type, then mark
pass/fail against the measured numbers.

| Document type | Target field acc | Measured (default provider) | Pass? |
| :-- | :-- | :-- | :-- |
| nin_slip | | | |
| passport | | | |
| drivers_license | | | |
| bank_mandate | | | |

## Rationale

- Why this default (accuracy vs cost vs latency trade-off):
- Why a two-provider strategy was / wasn't chosen:
- License / data-residency confirmation (see `LICENSE_AND_DEPLOYMENT_NOTES.md`):

## Risks & follow-ups

- [ ] Confirm exact license on pinned model revision
- [ ] Re-run with real, legally-cleared Nigerian samples
- [ ] Add reference transcripts to score raw `extract_text`
- [ ] Validate portrait extraction on real ID photos
- [ ] Set production latency/throughput SLOs

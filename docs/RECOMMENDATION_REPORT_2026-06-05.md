# Provider Recommendation — 2026-06-05

Filled-in recommendation per the spec's *Open Question #1* and *QA And
Benchmarking* sections, backed by the benchmark evidence at
`benchmarks/document-ocr/results/combined_real_multi/`.

## Decision

- **Default provider: `olmocr` (allenai/olmOCR-2-7B-1025, Apache-2.0).**
  Wins on every measured axis — field accuracy, required-field recall,
  classification, and schema-following — across all three document types in
  the evaluation set. Zero errors across the run.
- **Per-operation strategy:**
  - `classify` → **`olmocr`** (1.000 vs 0.625 Qwen-VL vs 0.125 GLM-OCR vs 0.500 Tesseract)
  - `extract_fields` → **`olmocr`** default; **`glm-ocr`** in latency-sensitive
    flows for `passport` and `nin_slip` (see §"Two-tier strategy" below)
  - `extract_text` → **`olmocr`** (proxied by field_accuracy here — see caveat)
  - `extract_portrait` → **shared face-detector** (all candidates tied at 0.875
    success, 0.523 mean quality; the VLMs all delegate to the OpenCV Haar
    cascade in `providers/portrait.py`)
- **Fallback chain (per the spec's provenance-preserving rule):**
  `olmocr → qwen-vl → glm-ocr → tesseract`
- **Dropped from candidate set:** `paddleocr-vl` — see §"Rejected candidates."
- **Status: provisional.** n=8 is enough to lock the top of the ranking but
  not enough to defend the smaller margins. Re-run on the expanded set
  before locking the production default (§"Required follow-ups").

## Evidence

- **Run id:** `combined_real_multi`
- **Dataset:** `real_multi_v1` — 8 real Nigerian documents
  (2 × drivers_license, 3 × nin_slip, 3 × passport), all `clean` capture condition,
  human-verified ground truth.
- **Hardware:** Coolify host, Linux x86_64, CPU-only (no GPU available for this run).
  Latency dominates because of this — see §"Latency reality."
- **Provider runs:** GLM-OCR, Qwen-VL, olmOCR, plus PaddleOCR-VL and Tesseract
  baseline.
- **Per-run evidence:** `benchmarks/document-ocr/results/combined_real_multi/`
  — `report.md`, `summary.json`, `results.json`.

### Composite ranking

| Rank | Candidate | Composite | Field acc | Req. recall | Classify | Schema | Portrait | p50 | p95 | Errors | Cost | License |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| 1 | `olmocr` | **0.791** | 0.850 | 0.842 | 1.000 | 1.000 | 0.875 | 811 s | 1 038 s | 0 % | medium | approved |
| 2 | `qwen-vl` | 0.636 | 0.668 | 0.583 | 0.625 | 1.000 | 0.875 | 707 s | 748 s | 0 % | high | approved |
| 3 | `glm-ocr` | 0.596 | 0.641 | 0.558 | 0.125 | 0.875 | 0.875 | 82 s | 634 s | 0 % | low | approved |
| 4 | `tesseract` | 0.337 | 0.075 | 0.042 | 0.500 | 0.000 | 0.875 | 2.6 s | 3.7 s | 0 % | low | approved |
| 5 | `paddleocr-vl` | 0.315 | 0.256 | 0.250 | 0.000 | 0.000 | 0.875 | 125 s | 765 s | 0 % | low | approved |

### Field accuracy by document type

| Candidate | drivers_license (n=2) | nin_slip (n=3) | passport (n=3) |
| :-- | --: | --: | --: |
| `olmocr` | **0.850** | **0.792** | **0.909** |
| `qwen-vl` | 0.711 | 0.678 | 0.631 |
| `glm-ocr` | 0.354 | 0.619 | 0.855 |
| `paddleocr-vl` | 0.064 | 0.640 | 0.000 |
| `tesseract` | 0.000 | 0.189 | 0.011 |

## Per-document-type acceptance check

Targets set at the start of the spike. Status read against the
**default-provider (olmOCR)** measurement.

| Document type | Target field acc | Measured (olmocr) | Pass? |
| :-- | --: | --: | :-- |
| `nin_slip` | ≥ 0.85 | 0.792 | **borderline FAIL** — within sample noise of target; rerun |
| `passport` | ≥ 0.85 | 0.909 | **PASS** |
| `drivers_license` | ≥ 0.85 | 0.850 | **borderline PASS** — at target, low sample size |

## Two-tier strategy (optional but supported by the evidence)

The spec explicitly permits "different defaults for `extract_text`,
`extract_fields`, `classify`, and `extract_portrait`" and a two-provider strategy
"if one model is better for field extraction and another is better for portrait."
This dataset supports a *cost/latency* split:

| Tier | Provider | Where it wins |
| :-- | :-- | :-- |
| **Accuracy-first** (default) | `olmocr` | All three doc types — best field accuracy, perfect classify + schema-following. |
| **Latency-first** (quick check) | `glm-ocr` | `passport` (0.855 vs 0.909 = -5.4 pp at **10× the throughput**: 82 s p50 vs 811 s); acceptable for `nin_slip` (0.619 vs 0.792 = -17.3 pp — only if latency dominates). |

Note: **GLM-OCR's classify is broken** in this dataset (0.125). The two-tier
mode therefore requires *fixing the document type out-of-band* (via the upload
endpoint hint or a separate classifier call) before routing to GLM-OCR.
`drivers_license` is **not eligible** for the GLM-OCR fast path — the
accuracy gap is too large (0.354 vs 0.850).

## Latency reality (CPU)

| Provider | p50 | p95 | Tier |
| :-- | --: | --: | :-- |
| `tesseract` | 2.6 s | 3.7 s | real-time, weak accuracy |
| `glm-ocr` | **82 s** | **634 s** | borderline sync; **note the 7.7× p95/p50 spread** — one slow tail |
| `paddleocr-vl` | 125 s | 765 s | async-only, weak accuracy |
| `qwen-vl` | 707 s | 748 s | async-only |
| `olmocr` | 811 s | 1 038 s | async-only |

On CPU, only Tesseract is real-time. To serve any of the VLMs interactively
**you need a GPU**; on GPU all three drop to ~1–3 s/doc and the "tier" column
becomes irrelevant.

GLM-OCR's tail-latency spread (p95 = 7.7 × p50) is worth attention before
committing to it as the fast lane: at the 95th percentile its latency is
within the same order of magnitude as olmOCR's median. The mean masks a
slow-tail behaviour, possibly the multi-page or denser pages.

## Rejected candidates

### PaddleOCR-VL — measured, not-suitable

- Field accuracy 0.256 (#4), driven by **0.000 on passport** and 0.064 on
  drivers_license. Only competitive on `nin_slip` (0.640).
- Classification 0.000, schema-following 0.000 (it returns markdown OCR text by
  design, not structured JSON).
- The harness's text-then-parse adapter (`text_kie.py`) extracts the labels it
  can find but the model's column-interleaved reading order on multi-column
  layouts (passport, DL) breaks label/value pairing.
- Honest characterisation: "an OCR engine that happens to read multilingual
  text well, but not a KIE model." Suitable as a `extract_text`-only baseline
  on a GPU host where its low cost is meaningful, **not** as the field
  extractor.

### Tesseract — fallback floor only

- Field accuracy 0.075 across the three types confirms what the spec already
  anticipates: "weak layout reasoning and structured field extraction."
- Useful as the local-only fallback for `extract_text` and as the latency floor
  for benchmark comparisons; never the field extractor.

## On-prem & licensing posture

All three top candidates are commercially-friendly and self-hostable:

| Provider | License | Self-hostable | Verified |
| :-- | :-- | :-- | :-- |
| `olmocr` (allenai/olmOCR-2-7B-1025) | Apache-2.0 | ✅ (vLLM / SGLang / Ollama) | confirmed via `providers/olmocr.py` and HF model card |
| `qwen-vl` (Qwen3-VL family) | Apache-2.0 | ✅ (vLLM ≥0.11 / SGLang) | confirmed |
| `glm-ocr` (zai-org/GLM-OCR) | MIT (weights); PP-DocLayoutV3 Apache-2.0 | ✅ (vLLM / SGLang / Ollama) | confirmed |

Licensing does **not** eliminate any candidate at this stage — the decision
rests entirely on accuracy and latency. **Confirm the exact `LICENSE` file on
the pinned model revision before a production customer rollout** (these
projects move fast).

## Risks & follow-ups

- [ ] **Expand the evaluation set to ~10 per type (30 docs total)** before
  locking the production default. The current per-type cells rest on n=2/3 —
  enough to detect large gaps (olmOCR vs PaddleOCR-VL on passport: ~91 pp)
  but not enough to defend smaller ones (olmOCR vs Qwen-VL on drivers_license:
  ~14 pp).
- [ ] **Re-run on a GPU host (`docker-compose.gpu.yml`).** Composite ranking
  will hold but the latency sub-score stops dominating, which lets per-type
  accuracy drive the decision more cleanly. Also collects real GPU-memory
  numbers for the on-prem cost story.
- [ ] **Pin model revisions and re-verify licenses** before customer rollout.
- [ ] **Add reference transcripts** so `extract_text` is scored directly rather
  than via field-accuracy proxy.
- [ ] **Add capture-condition variants** of the real set (mobile, glare,
  low-light, rotated, WhatsApp) once the clean baseline holds — only `clean`
  is measured here, so the spec's robustness dimension is unmeasured for real
  docs.
- [ ] **Investigate GLM-OCR's classification failure** (0.125). Either fix the
  prompt or accept that GLM-OCR is "fields-only, route by external classifier"
  in the two-tier strategy.
- [ ] **Investigate GLM-OCR's p95 latency spread** before committing it as the
  fast lane.
- [ ] **Add olmOCR to the harness's allowed-providers default** in
  `docker-compose.coolify.yml` (currently only the original four).

## What this answers from the spec's "Open Questions"

| Spec question | Answer |
| :-- | :-- |
| **1. Which two providers should be wired into the first benchmark spike?** | Wired all four (olmOCR, Qwen-VL, GLM-OCR, PaddleOCR-VL) plus Tesseract baseline. Measured: **olmOCR wins**; PaddleOCR-VL rejected. |
| **2. What real document samples can be legally used for benchmarking?** | Eight legally-cleared real Nigerian documents (drivers_license / nin_slip / passport), human-authored ground truth in `expected/real/`. |
| **3. What minimum benchmark score is acceptable per document type?** | Targets set at 0.85 field accuracy; measured: passport PASS, drivers_license borderline PASS, nin_slip borderline FAIL — rerun on expanded set. |
| **4. Should artifact storage be included in MVP?** | Deferred. Harness is stateless; managed-artifact mode designed but not built. |
| **5. Which product consumes OCR first?** | Open. Recommend Bifense IDV (the spec's primary use case) — the chosen olmOCR provider supports the document-portrait + field-extraction shape Bifense needs. |

---

*Drafted from `benchmarks/document-ocr/results/combined_real_multi/summary.json`
(40 result rows, 5 candidates, 8 samples, 0 errors). Composite weights and
latency budget at the time of run: `field_accuracy 0.35, required_recall 0.15,
classification 0.10, schema_following 0.10, latency 0.10, portrait 0.10, cost
0.05, license 0.05; latency_budget 8 000 ms (p50)`. See `report.py` for the
scoring rationale; weights are tunable.*

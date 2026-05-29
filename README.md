# Document OCR Engine

Nigeria-native document intelligence for banking, fintech, and identity
workflows — built as a standalone, provider-neutral OCR engine that Bifense,
Banklet, and standalone bank customers consume through a stable API and adapters.

See [`DOCUMENT_OCR_ENGINE_SERVICE_SPEC.md`](DOCUMENT_OCR_ENGINE_SERVICE_SPEC.md)
for the full target architecture.

## Where this repo is right now: the benchmark spike

The spec sets a hard sequencing rule:

> **Benchmark-First Delivery Rule** — Deep service development should not start
> before the VLM/OCR candidates are evaluated. The first implementation milestone
> is a benchmark spike, not a complete OCR service. The default provider must be
> chosen from measured results on our own document set plus license, deployment,
> and cost review.

So the **first deliverable is the benchmark harness**, and that is what this repo
currently implements:

- ✅ runnable Python environment + Docker image (Coolify-deployable)
- ✅ sample manifest format + ground-truth schema
- ✅ provider runner interface with capability reporting
- ✅ provider runners: **Qwen-VL, GLM-OCR, PaddleOCR-VL, olmOCR** (OpenAI-compatible/vLLM)
  and **Tesseract** (classical fallback, runs today)
- ✅ synthetic Nigerian specimen generator (8 doc types × capture conditions)
- ✅ scoring (field accuracy, required-field recall, classification, schema-following,
  portrait), resource measurement (latency, GPU memory), robustness by capture condition
- ✅ JSON evidence output + Markdown recommendation report with composite scoring
- ✅ per-candidate license/deployment notes

It does **not** yet implement the production FastAPI service, adapters, or async
jobs — those come *after* a provider is chosen from evidence (spec Implementation
Order, steps 6+).

## Why it's not "accuracy only"

The harness weighs raw OCR accuracy **alongside** latency, GPU cost, schema-following
reliability, portrait extraction, capture-condition robustness, and license/on-prem
posture. A model that is 1% more accurate at 10× the GPU cost can lose. Weights are
explicit and tunable in [`report.py`](packages/document-ocr-benchmarks/src/document_ocr_benchmarks/report.py).

## Quick start (local, CPU)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e "packages/document-ocr-benchmarks[tesseract,dev]"

document-ocr-bench gen-samples --out benchmarks/document-ocr   # synthetic dataset
document-ocr-bench providers                                   # candidates + license
document-ocr-bench run --providers tesseract                   # run + build report
```

The report lands in `benchmarks/document-ocr/results/<run_id>/report.md`.

## Quick start (Docker / Coolify — CPU or GPU)

```bash
cp .env.example .env                       # VLLM_IMAGE, HUGGING_FACE_HUB_TOKEN, endpoints
docker compose --profile glm up -d glm-vllm       # serve a candidate (CPU by default)
docker compose run --rm bench run --providers glm-ocr,tesseract
```

Model servers use official [vLLM CPU images](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags)
by default (no NVIDIA required). For GPU hosts, add `-f docker-compose.gpu.yml`.
The `bench` container orchestrates and scores; VLMs are reached over OpenAI-compatible
HTTP. Full instructions in [`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md).

## Repository layout

```
packages/document-ocr-benchmarks/   # the harness (installable Python package)
  src/document_ocr_benchmarks/
    models.py            provider-neutral data models + benchmark evidence record
    schemas.py           Nigerian field schemas per document type
    normalization.py     field normalization + NG validation rules
    imaging.py           image loading + deterministic quality assessment
    scoring.py           ground-truth scoring
    resources.py         latency / RSS / GPU memory measurement
    harness.py           benchmark orchestrator
    report.py            aggregation, composite score, recommendation
    cli.py               document-ocr-bench CLI
    providers/           base, registry, portrait (face detector),
                         vlm_openai, qwen_vl, glm_ocr, paddleocr_vl, olmocr,
                         tesseract, mock
    synth/               synthetic specimen generator + degradations + faces
  tests/                 unit + end-to-end pipeline tests

benchmarks/document-ocr/            # the dataset + results (the spike's data dir)
  manifests/  samples/  expected/  results/  scripts/

docs/                               # license notes, benchmark guide, report template
Dockerfile  docker-compose.yml  .env.example
```

## Open questions the spike is set up to answer

These map to the spec's "Open Questions For Implementation Planning":

1. **Which providers in the first benchmark?** — wired: Qwen-VL, GLM-OCR,
   PaddleOCR-VL, Tesseract (olmOCR also available).
2. **What samples?** — synthetic specimen generator now; same format accepts real
   legally-cleared samples (see the guide).
3. **Minimum acceptable score per document type?** — record in
   `docs/RECOMMENDATION_REPORT_TEMPLATE.md` after the first real-sample run.
4. **Artifact storage in MVP?** — deferred; the harness is stateless.
5. **Which product consumes OCR first?** — open; harness output is product-neutral.

## License

Apache-2.0.

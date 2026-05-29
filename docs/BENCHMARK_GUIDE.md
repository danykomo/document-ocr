# Benchmark Guide

How to run the spike on a Coolify or Docker host (CPU or GPU), serve the VLM
candidates, and slot in real Nigerian document samples.

## 1. Run locally (CPU, classical only)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e "packages/document-ocr-benchmarks[tesseract,dev]"

document-ocr-bench gen-samples --out benchmarks/document-ocr   # synthetic data
document-ocr-bench providers                                   # what's available
document-ocr-bench run --providers tesseract                   # run + report
```

Outputs land in `benchmarks/document-ocr/results/<run_id>/`:
`results.json`, `results.jsonl`, `run_meta.json`, `summary.json`, `report.md`.

## 2. Serve a VLM candidate (CPU or GPU)

The harness reaches every VLM over an OpenAI-compatible endpoint, so any of
vLLM / SGLang works.

### CPU (default — no NVIDIA required)

`docker-compose.yml` uses the official [vLLM CPU images](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags).
Copy `.env.example` to `.env` and set `VLLM_IMAGE` for your platform:

- Linux **x86_64**: `vllm/vllm-openai-cpu:latest-x86_64` (default)
- Linux **arm64** / **Apple Silicon**: `vllm/vllm-openai-cpu:latest-arm64`

```bash
cp .env.example .env
export HUGGING_FACE_HUB_TOKEN=hf_...
docker compose --profile glm up -d glm-vllm       # serves GLM-OCR on :18002
docker compose --profile qwen up -d qwen-vllm     # serves Qwen3-VL-8B on :18001
```

CPU inference is memory-hungry and slower than GPU. Start with the smallest
models (`glm`, `paddle`) before `qwen` or `olmocr`. Tune `VLLM_CPU_KVCACHE_SPACE`
and `OCR_SYNC_TIMEOUT_MS` if requests time out.

### GPU (optional, faster)

If the host has an NVIDIA GPU and the Container Toolkit installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile qwen up -d qwen-vllm
```

### Manual vLLM

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000 --trust-remote-code --device cpu
```

Then point the harness at it and run:

```bash
export OCR_PROVIDER_QWEN_VL_ENDPOINT=http://localhost:8000/v1
export OCR_PROVIDER_QWEN_VL_MODEL=Qwen/Qwen3-VL-8B-Instruct
document-ocr-bench run --providers qwen-vl,glm-ocr,paddleocr-vl,tesseract
```

GPU memory is sampled via `nvidia-smi` when the harness can see a GPU. On CPU-only
hosts, or when the harness calls a remote endpoint, the report GPU column shows `—`.

## 3. Run the whole stack via Docker (Coolify)

1. Deploy this repo as a **Docker Compose** application (any CPU-capable host; GPU optional).
2. Set env from `.env.example` (`HUGGING_FACE_HUB_TOKEN`, `VLLM_IMAGE`, provider endpoints).
3. Enable the compose profile(s) you need, e.g. `COMPOSE_PROFILES=glm`.
4. Bring up the model server(s), wait for weights to download and the server to listen.
5. Run the harness as a one-shot task:
   ```bash
   docker compose run --rm bench run --providers glm-ocr,tesseract
   ```
6. Collect `benchmarks/document-ocr/results/<run_id>/report.md`.

## 4. Add real (legally-cleared) samples

The harness treats synthetic and real samples identically. To add real samples:

1. Drop images in `benchmarks/document-ocr/samples/<sample_id>.jpg`.
2. Add a ground-truth file `benchmarks/document-ocr/expected/<sample_id>.json`:
   ```json
   {
     "sample_id": "real-passport-001",
     "document_type": "passport",
     "fields": { "full_name": "...", "passport_number": "A01234567", "...": "..." },
     "required_fields": ["full_name", "passport_number", "expiry_date"],
     "portrait_present": true
   }
   ```
3. Add an entry to a manifest under `benchmarks/document-ocr/manifests/` (or copy
   the synthetic manifest and append). Each entry needs `sample_id`,
   `document_type`, `image_path`, and `capture_condition`.
4. Validate and run:
   ```bash
   document-ocr-bench validate --manifest benchmarks/document-ocr/manifests/real_v1.json
   document-ocr-bench run --manifest benchmarks/document-ocr/manifests/real_v1.json
   ```

> Never commit real documents containing PII. `.gitignore` excludes `samples/*.jpg`
> by default; keep cleared samples in secure storage and mount them at run time.

## 5. Reading the report

- **Composite score** weighs accuracy with latency, cost, license, schema-following,
  and portrait — tune the weights in `report.py`.
- **Per-operation winners** support a two-provider strategy (e.g. one model for
  `extract_fields`, another for `extract_portrait`).
- **By document type** shows where a model is strong/weak (set per-type acceptance
  thresholds here).
- **By capture condition** is the robustness view (mobile, glare, rotated, WhatsApp).

## Scoring notes

- Numbers (NIN/BVN/passport) and dates are scored **strictly** (exact after
  normalization). Names/addresses get fuzzy partial credit.
- A required field counts as recalled at ≥ 0.85 score.
- `extract_text` currently uses field accuracy as a proxy; add reference
  transcripts to score raw OCR directly.

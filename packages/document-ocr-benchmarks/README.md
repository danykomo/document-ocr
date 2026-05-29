# document-ocr-benchmarks

Benchmark harness for evaluating VLM/OCR providers (GLM-OCR, Qwen-VL,
PaddleOCR-VL, olmOCR, Tesseract fallback) on Nigerian banking and identity
documents.

This package is the **first deliverable** of the Document OCR Engine: per the
service spec's *Benchmark-First Delivery Rule*, the default provider must be
chosen from measured evidence on our own document set — not public leaderboards.

It measures more than raw OCR accuracy:

- key-field extraction accuracy + required-field recall
- document classification accuracy
- portrait extraction (face-detector based)
- deterministic JSON / schema-following reliability
- latency and (when a GPU is visible) GPU memory
- robustness across capture conditions (clean, mobile, low-light, glare, rotated, WhatsApp-compressed)
- commercial license + on-prem posture per candidate

See the repository root `README.md` and `docs/` for full usage, Docker, and
deployment instructions.

## Quick start

```bash
pip install -e ".[tesseract,dev]"
document-ocr-bench gen-samples --out benchmarks/document-ocr
document-ocr-bench providers
document-ocr-bench run --providers tesseract
```

# benchmarks/document-ocr

The benchmark dataset and results live here. Layout:

```
manifests/   # <name>.json — Manifest listing samples (synthetic + real)
samples/     # <sample_id>.jpg — document images (gitignored by default)
expected/    # <sample_id>.json — ExpectedFields ground truth per sample
results/     # <run_id>/ — results.json, results.jsonl, run_meta.json,
             #             summary.json, report.md  (gitignored)
scripts/     # ad-hoc helpers
```

## Regenerate the synthetic dataset

```bash
document-ocr-bench gen-samples --out benchmarks/document-ocr
```

Synthetic documents are clearly watermarked `SPECIMEN - NOT A REAL DOCUMENT` and
carry randomly-generated (fictional) field values. They exist so the harness is
runnable today with zero legal/privacy risk and so scoring has known ground truth.

## Add real samples

Real, legally-cleared samples use the **same format** — see
[`docs/BENCHMARK_GUIDE.md`](../../docs/BENCHMARK_GUIDE.md). Never commit images
containing PII; `.gitignore` excludes `samples/*.jpg` by default.

## Capture conditions

Each base document is emitted under several capture conditions to measure
robustness: `clean`, `mobile`, `low_light`, `glare`, `rotated`, `whatsapp`
(and `blurred`, `cropped`, `photocopy` are available in the generator).

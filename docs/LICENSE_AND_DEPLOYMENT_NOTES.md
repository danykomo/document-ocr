# Candidate License & Deployment Notes

Per the spec's selection rule, a candidate is rejected if it has an unacceptable
commercial-license or data-residency posture **even if its benchmark score is
high**. This document captures the license and deployment facts so that gate can
be applied alongside the measured benchmark evidence.

> Verified against public model cards / repos as of **2026-05**. Re-verify the
> exact `LICENSE` file on the pinned model revision before a production rollout —
> these projects move fast.

## Summary

| Candidate | Params | License | Commercial use | On-prem | Serving | Cost tier |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| GLM-OCR | ~0.9B | MIT (weights); PP-DocLayoutV3 component Apache-2.0 | ✅ permissive | ✅ | vLLM / SGLang / Ollama / Transformers | low |
| Qwen3-VL | 2B–235B | Apache-2.0 | ✅ permissive | ✅ | vLLM ≥0.11 / SGLang (OpenAI-compatible) | high (large), med (2–8B) |
| PaddleOCR-VL | ~0.9B | Apache-2.0 | ✅ permissive | ✅ | vLLM (officially supported) | low |
| olmOCR 2 | 7B | Apache-2.0 | ✅ permissive | ✅ | vLLM / SGLang / Ollama / LM Studio | medium |
| Tesseract | n/a | Apache-2.0 | ✅ permissive | ✅ | local CPU library | low |

**Key finding:** every candidate is MIT/Apache-2.0 and fully self-hostable, so
the on-prem / data-residency requirement (Nigerian banks keeping documents in
their own infrastructure) is satisfiable by all of them. The decision therefore
hinges on **measured accuracy on Nigerian documents, latency, and GPU cost** —
not licensing. Licensing does not eliminate any candidate at this stage.

---

## GLM-OCR (`zai-org/GLM-OCR`)

- **Role:** compact OCR-specialized contender; the spec explicitly wants it in
  the first set.
- **Architecture:** ~0.9B (CogViT visual encoder + GLM-0.5B decoder), Multi-Token
  Prediction, two-stage layout parsing.
- **License:** model weights MIT. The full document pipeline bundles
  **PP-DocLayoutV3 (Apache-2.0)** — comply with both. Both are commercial-friendly.
- **Serving:** vLLM, SGLang, Ollama; Transformers support exists. Edge-deployable
  due to small size. Fine-tuning via LLaMA-Factory.
- **Strengths to confirm:** document parsing + KIE quality at a tiny footprint
  (strong public OmniDocBench / KIE numbers for its size).
- **Risks to validate:** runtime maturity; **portrait behavior** (it is an OCR
  model, not an image cropper — portrait extraction is delegated to the shared
  face detector in this harness); deterministic JSON field output.

## Qwen3-VL (`Qwen/Qwen3-VL-*`)

- **Role:** high-accuracy general VLM contender.
- **Variants:** 2B / 4B / 8B / 32B Instruct & Thinking, plus MoE 30B-A3B and
  235B-A22B; FP8 quants available. **8B Instruct** is the suggested on-prem default.
- **License:** Apache-2.0.
- **Serving:** vLLM ≥0.11 or SGLang, both OpenAI-compatible. Strong, flexible
  JSON extraction and document understanding; OCR robust to low light/blur/tilt.
- **Risks to validate:** GPU cost and latency on larger variants; deterministic
  field output (constrain with low temperature + strict prompt, as this harness does).

## PaddleOCR-VL (`PaddlePaddle/PaddleOCR-VL`, and `-1.5`)

- **Role:** efficient production baseline.
- **Architecture:** ~0.9B (NaViT dynamic-resolution encoder + ERNIE-4.5-0.3B),
  109 languages, strong layout/table/formula/chart parsing.
- **License:** Apache-2.0.
- **Serving:** officially supported on vLLM (OpenAI-compatible). Low serving cost.
- **Risks to validate:** **field-level** extraction quality on Nigerian IDs and
  bank forms vs general VLMs (it is tuned for document *parsing*/layout).

## olmOCR 2 (`allenai/olmOCR-2-7B-1025`)

- **Role:** open, reproducible reference pipeline.
- **Architecture:** 7B, fine-tuned from Qwen2.5-VL-7B; RL-tuned on tables/equations.
- **License:** Apache-2.0.
- **Serving:** vLLM / SGLang / Ollama / LM Studio; FP8 variant ~8.85 GB.
- **Risks to validate:** tuned for English print/PDF *linearization* — structured
  field extraction and portrait extraction need extra components; less proven on
  Nigerian IDs. Useful as a transparency/reproducibility reference.

## Tesseract (classical fallback)

- **Role:** fallback baseline / local-only floor.
- **License:** Apache-2.0. CPU-only, no model download, fully offline.
- **Risks:** weak layout reasoning and structured field extraction (confirmed in
  the harness: high on simple label-value bills, low on structured IDs). Use as a
  fallback in the chain, never as the primary field extractor.

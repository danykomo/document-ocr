# Benchmark harness image.
# CPU-only by design: the harness orchestrates, scores, and reports. The heavy
# VLM candidates (Qwen-VL, GLM-OCR, PaddleOCR-VL, olmOCR) are served separately
# via vLLM (see docker-compose.yml; CPU or optional GPU) over OpenAI-compatible
# HTTP, so this image stays small and deploys cleanly on Coolify.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# tesseract = the classical fallback runner; libgl/glib = OpenCV runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package first (better layer caching) then copy the rest.
COPY packages/document-ocr-benchmarks/pyproject.toml packages/document-ocr-benchmarks/README.md ./packages/document-ocr-benchmarks/
COPY packages/document-ocr-benchmarks/src ./packages/document-ocr-benchmarks/src
RUN pip install "./packages/document-ocr-benchmarks[tesseract]"

COPY benchmarks ./benchmarks

# Default: print help. Override with a `run`/`gen-samples` command, e.g.
#   docker run --rm -v $PWD/benchmarks:/app/benchmarks ocr-bench run --providers qwen-vl
ENTRYPOINT ["document-ocr-bench"]
CMD ["--help"]

"""Runtime configuration.

Environment variable names match the service spec (``OCR_*``) so the benchmark
harness and the eventual service share one configuration vocabulary. Provider
endpoints/keys are read here and passed to the HTTP-based VLM runners.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderEndpoint(BaseSettings):
    """Connection details for one OpenAI-compatible VLM endpoint (e.g. vLLM)."""

    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None


class Settings(BaseSettings):
    """Harness + provider configuration, sourced from env / .env."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # --- General service-shaped knobs (spec: Configuration) --------------- #
    default_provider: str = "tesseract"
    allowed_providers: list[str] = Field(
        default_factory=lambda: ["qwen-vl", "glm-ocr", "paddleocr-vl", "tesseract"]
    )
    local_only_mode: bool = False
    max_image_bytes: int = 15 * 1024 * 1024
    sync_timeout_ms: int = 60_000
    debug_output_enabled: bool = False

    # --- Provider endpoints (OpenAI-compatible, served via vLLM/SGLang) --- #
    # Read as e.g. OCR_PROVIDER_QWEN_VL_ENDPOINT / _API_KEY / _MODEL
    provider_qwen_vl_endpoint: Optional[str] = None
    provider_qwen_vl_api_key: Optional[str] = None
    provider_qwen_vl_model: str = "Qwen/Qwen3-VL-8B-Instruct"

    provider_glm_ocr_endpoint: Optional[str] = None
    provider_glm_ocr_api_key: Optional[str] = None
    provider_glm_ocr_model: str = "zai-org/GLM-OCR"

    provider_paddleocr_vl_endpoint: Optional[str] = None
    provider_paddleocr_vl_api_key: Optional[str] = None
    provider_paddleocr_vl_model: str = "PaddlePaddle/PaddleOCR-VL"

    provider_olmocr_endpoint: Optional[str] = None
    provider_olmocr_api_key: Optional[str] = None
    provider_olmocr_model: str = "allenai/olmOCR-2-7B-1025"

    # --- Classical fallback ---------------------------------------------- #
    provider_tesseract_enabled: bool = True
    provider_paddleocr_enabled: bool = False
    tesseract_lang: str = "eng"

    # --- Resource sampling ------------------------------------------------ #
    # Sample local nvidia-smi for GPU memory when a GPU is present (optional;
    # CPU-only hosts leave this enabled — readings are simply omitted in reports).
    gpu_query_enabled: bool = True

    def endpoint_for(self, provider: str) -> ProviderEndpoint:
        mapping = {
            "qwen-vl": ProviderEndpoint(
                endpoint=self.provider_qwen_vl_endpoint,
                api_key=self.provider_qwen_vl_api_key,
                model=self.provider_qwen_vl_model,
            ),
            "glm-ocr": ProviderEndpoint(
                endpoint=self.provider_glm_ocr_endpoint,
                api_key=self.provider_glm_ocr_api_key,
                model=self.provider_glm_ocr_model,
            ),
            "paddleocr-vl": ProviderEndpoint(
                endpoint=self.provider_paddleocr_vl_endpoint,
                api_key=self.provider_paddleocr_vl_api_key,
                model=self.provider_paddleocr_vl_model,
            ),
            "olmocr": ProviderEndpoint(
                endpoint=self.provider_olmocr_endpoint,
                api_key=self.provider_olmocr_api_key,
                model=self.provider_olmocr_model,
            ),
        }
        return mapping.get(provider, ProviderEndpoint())


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

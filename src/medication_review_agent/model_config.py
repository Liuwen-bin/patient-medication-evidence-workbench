from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class ModelConfigurationError(ValueError):
    pass


@dataclass(frozen=True, repr=False)
class LLMSettings:
    base_url: str
    api_key: SecretStr
    model: str
    timeout_seconds: float = 30.0
    max_retries: int = 0

    def __repr__(self) -> str:
        return f"LLMSettings(base_url={self.base_url!r}, model={self.model!r})"


def load_llm_settings(environ: Mapping[str, str] | None = None) -> LLMSettings:
    values = dict(os.environ if environ is None else environ)
    source_path = values.get("AGENT_MODEL_ENV_PATH", "").strip()
    file_values = dotenv_values(source_path) if source_path else {}

    def pick(*keys: str) -> str:
        for key in keys:
            for source in (values, file_values):
                normalized = str(source.get(key) or "").strip()
                if normalized:
                    return normalized
        return ""

    base_url = pick(
        "AGENT_LLM_BASE_URL",
        "QUERY_LLM_BINDING_HOST",
        "LLM_BINDING_HOST",
    )
    api_key = pick(
        "AGENT_LLM_API_KEY",
        "QUERY_LLM_BINDING_API_KEY",
        "LLM_BINDING_API_KEY",
    )
    model = pick("AGENT_LLM_MODEL", "QUERY_LLM_MODEL", "LLM_MODEL")
    if not base_url or not api_key or not model:
        raise ModelConfigurationError("Model base URL and model name must be configured")

    try:
        timeout_seconds = float(values.get("AGENT_LLM_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        raise ModelConfigurationError("Model timeout must be a number") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ModelConfigurationError("Model timeout must be greater than zero")

    return LLMSettings(
        base_url=base_url.rstrip("/"),
        api_key=SecretStr(api_key),
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=0,
    )


def build_chat_model(settings: LLMSettings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        temperature=0,
    )

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medication_review_agent import model_config
from medication_review_agent.model_config import (
    ModelConfigurationError,
    build_chat_model,
    load_llm_settings,
)


def test_query_binding_keys_take_precedence(tmp_path: Path) -> None:
    source = tmp_path / "model.env"
    source.write_text(
        "QUERY_LLM_BINDING_HOST=https://model.example/v1\n"
        "LLM_BINDING_HOST=https://fallback.example/v1\n"
        "QUERY_LLM_BINDING_API_KEY=secret-query\n"
        "LLM_BINDING_API_KEY=secret-fallback\n"
        "QUERY_LLM_MODEL=model-query\n"
        "LLM_MODEL=model-fallback\n",
        encoding="utf-8",
    )

    settings = load_llm_settings({"AGENT_MODEL_ENV_PATH": str(source)})

    assert settings.base_url == "https://model.example/v1"
    assert settings.model == "model-query"
    assert settings.api_key.get_secret_value() == "secret-query"
    assert "secret-query" not in repr(settings)


def test_agent_process_settings_override_external_file(tmp_path: Path) -> None:
    source = tmp_path / "model.env"
    source.write_text(
        "QUERY_LLM_BINDING_HOST=https://file.example/v1\n"
        "QUERY_LLM_BINDING_API_KEY=file-secret\n"
        "QUERY_LLM_MODEL=file-model\n",
        encoding="utf-8",
    )

    settings = load_llm_settings({
        "AGENT_MODEL_ENV_PATH": str(source),
        "AGENT_LLM_BASE_URL": "https://process.example/v1/",
        "AGENT_LLM_API_KEY": "process-secret",
        "AGENT_LLM_MODEL": "process-model",
        "AGENT_LLM_TIMEOUT_SECONDS": "12.5",
    })

    assert settings.base_url == "https://process.example/v1"
    assert settings.api_key.get_secret_value() == "process-secret"
    assert settings.model == "process-model"
    assert settings.timeout_seconds == 12.5


def test_whitespace_setting_does_not_block_valid_fallback(tmp_path: Path) -> None:
    source = tmp_path / "model.env"
    source.write_text(
        "QUERY_LLM_BINDING_HOST=https://file.example/v1\n"
        "QUERY_LLM_BINDING_API_KEY=file-secret\n"
        "QUERY_LLM_MODEL=file-model\n",
        encoding="utf-8",
    )

    settings = load_llm_settings({
        "AGENT_MODEL_ENV_PATH": str(source),
        "AGENT_LLM_MODEL": "   ",
    })

    assert settings.model == "file-model"


def test_external_file_loading_does_not_mutate_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.env"
    source.write_text(
        "QUERY_LLM_BINDING_HOST=https://model.example/v1\n"
        "QUERY_LLM_BINDING_API_KEY=external-secret\n"
        "QUERY_LLM_MODEL=external-model\n",
        encoding="utf-8",
    )
    keys = (
        "QUERY_LLM_BINDING_HOST",
        "QUERY_LLM_BINDING_API_KEY",
        "QUERY_LLM_MODEL",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    load_llm_settings({"AGENT_MODEL_ENV_PATH": str(source)})

    assert all(key not in os.environ for key in keys)


def test_missing_model_setting_names_safe_fields_only(tmp_path: Path) -> None:
    source = tmp_path / "empty.env"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ModelConfigurationError) as error:
        load_llm_settings({"AGENT_MODEL_ENV_PATH": str(source)})

    assert "API_KEY" not in str(error.value)
    assert "base URL and model name" in str(error.value)


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf", "0", "-1"])
def test_timeout_must_be_positive_and_finite(timeout: str) -> None:
    with pytest.raises(ModelConfigurationError, match="greater than zero"):
        load_llm_settings({
            "AGENT_LLM_BASE_URL": "https://model.example/v1",
            "AGENT_LLM_API_KEY": "secret",
            "AGENT_LLM_MODEL": "model",
            "AGENT_LLM_TIMEOUT_SECONDS": timeout,
        })


def test_chat_model_disables_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_chat_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(model_config, "ChatOpenAI", fake_chat_openai)
    settings = load_llm_settings({
        "AGENT_LLM_BASE_URL": "https://model.example/v1",
        "AGENT_LLM_API_KEY": "secret",
        "AGENT_LLM_MODEL": "model",
    })

    build_chat_model(settings)

    assert captured["max_retries"] == 0
    assert captured["temperature"] == 0
    assert captured["timeout"] == 30.0

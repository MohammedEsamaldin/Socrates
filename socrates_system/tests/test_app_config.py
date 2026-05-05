"""
Tests for the AppConfig / get_app_config() machinery.

Each test that mutates env vars resets the lru_cache and restores the
original environment so tests remain independent.
"""
from __future__ import annotations

import os
import pytest

from socrates_system.config import AppConfig, get_app_config


def _fresh_config(**env_overrides) -> AppConfig:
    """Return a fresh AppConfig after temporarily applying *env_overrides*."""
    original = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v
    get_app_config.cache_clear()
    try:
        return get_app_config()
    finally:
        # Restore env and reset cache so later tests get a clean slate
        for k, orig_v in original.items():
            if orig_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig_v
        get_app_config.cache_clear()


class TestAppConfigDefaults:
    def test_returns_app_config_instance(self):
        get_app_config.cache_clear()
        cfg = get_app_config()
        assert isinstance(cfg, AppConfig)

    def test_llm_provider_default(self):
        cfg = _fresh_config()
        assert cfg.llm_provider == "ollama"

    def test_ollama_base_url_default(self):
        cfg = _fresh_config()
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_factuality_enabled_default(self):
        cfg = _fresh_config()
        assert cfg.factuality_enabled is True

    def test_clarification_enabled_default(self):
        cfg = _fresh_config()
        assert cfg.clarification_enabled is True

    def test_allow_polarity_flip_default(self):
        cfg = _fresh_config()
        assert cfg.allow_polarity_flip is False

    def test_agla_api_url_default_empty(self):
        cfg = _fresh_config()
        assert cfg.agla_api_url == ""

    def test_qg_min_confidence_default_none(self):
        cfg = _fresh_config()
        assert cfg.qg_min_confidence is None

    def test_no_color_default_false(self):
        cfg = _fresh_config()
        assert cfg.no_color is False


class TestAppConfigEnvOverrides:
    def test_llm_provider_from_env(self):
        cfg = _fresh_config(SOC_LLM_PROVIDER="openai")
        assert cfg.llm_provider == "openai"

    def test_llm_model_from_env(self):
        cfg = _fresh_config(SOC_LLM_MODEL="gpt-4o")
        assert cfg.llm_model == "gpt-4o"

    def test_llm_model_empty_env_yields_none(self):
        cfg = _fresh_config(SOC_LLM_MODEL="")
        assert cfg.llm_model is None

    def test_factuality_disabled_via_env(self):
        cfg = _fresh_config(FACTUALITY_ENABLED="false")
        assert cfg.factuality_enabled is False

    def test_factuality_enabled_via_true_string(self):
        cfg = _fresh_config(FACTUALITY_ENABLED="true")
        assert cfg.factuality_enabled is True

    def test_agla_timeout_from_env(self):
        cfg = _fresh_config(AGLA_API_TIMEOUT="30")
        assert cfg.agla_api_timeout == 30

    def test_agla_timeout_bad_value_uses_default(self):
        cfg = _fresh_config(AGLA_API_TIMEOUT="notanumber")
        assert cfg.agla_api_timeout == 120

    def test_allow_polarity_flip_from_env(self):
        cfg = _fresh_config(SOC_ALLOW_POLARITY_FLIP="true")
        assert cfg.allow_polarity_flip is True

    def test_no_color_from_env(self):
        cfg = _fresh_config(NO_COLOR="1")
        assert cfg.no_color is True

    def test_qg_min_confidence_from_env(self):
        cfg = _fresh_config(QG_MIN_CONFIDENCE_THRESHOLD="0.65")
        assert cfg.qg_min_confidence == pytest.approx(0.65)

    def test_qg_min_confidence_bad_value_yields_none(self):
        cfg = _fresh_config(QG_MIN_CONFIDENCE_THRESHOLD="bad")
        assert cfg.qg_min_confidence is None

    def test_clarification_refine_questions_from_env(self):
        cfg = _fresh_config(REFINE_QUESTIONS_WITH_LLM="false")
        assert cfg.clarification_refine_questions is False

    def test_factuality_timeout_from_env(self):
        cfg = _fresh_config(FACTUALITY_TIMEOUT="12.5")
        assert cfg.factuality_timeout == pytest.approx(12.5)

    def test_ollama_base_url_from_env(self):
        cfg = _fresh_config(OLLAMA_BASE_URL="http://myhost:11434")
        assert cfg.ollama_base_url == "http://myhost:11434"

    def test_ollama_url_fallback(self):
        cfg = _fresh_config(OLLAMA_URL="http://fallback:11434")
        assert cfg.ollama_base_url == "http://fallback:11434"


class TestAppConfigCaching:
    def test_same_object_returned_on_second_call(self):
        get_app_config.cache_clear()
        a = get_app_config()
        b = get_app_config()
        assert a is b

    def test_cache_clear_allows_reload(self):
        get_app_config.cache_clear()
        os.environ["SOC_LLM_PROVIDER"] = "openai"
        try:
            cfg1 = get_app_config()
            assert cfg1.llm_provider == "openai"
            get_app_config.cache_clear()
            os.environ["SOC_LLM_PROVIDER"] = "ollama"
            cfg2 = get_app_config()
            assert cfg2.llm_provider == "ollama"
            assert cfg1 is not cfg2
        finally:
            os.environ.pop("SOC_LLM_PROVIDER", None)
            get_app_config.cache_clear()

    def test_config_is_frozen(self):
        get_app_config.cache_clear()
        cfg = get_app_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.llm_provider = "something_else"  # type: ignore[misc]

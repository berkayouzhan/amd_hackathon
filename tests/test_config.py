"""
test_config.py
===============
config modulu icin birim testleri.

Kontrol edilenler:
  - Zorunlu env degiskenleri eksik oldugunda ConfigError firlatirilmasi
  - ALLOWED_MODELS parsing'i
  - Rol otomatik atamasi (default/reasoning/code)
  - is_model_allowed guard'i
  - Gemma ayarlarinin yuklenmesi
"""

import pytest
from config import ConfigError, _parse_allowed_models, load_settings


class TestParseAllowedModels:
    """ALLOWED_MODELS string'inin dogru parse edilmesi."""

    def test_comma_separated(self):
        assert _parse_allowed_models("a,b,c") == ["a", "b", "c"]

    def test_with_spaces(self):
        assert _parse_allowed_models("a , b , c") == ["a", "b", "c"]

    def test_empty_string(self):
        assert _parse_allowed_models("") == []

    def test_single_model(self):
        assert _parse_allowed_models("accounts/fireworks/models/minimax-m3") == [
            "accounts/fireworks/models/minimax-m3"
        ]


class TestLoadSettings:
    """load_settings()'in env degiskenlerine gore dogru Settings donmesi."""

    def test_successful_load(self, settings):
        assert settings.fireworks_api_key == "fw_test_key"
        assert settings.fireworks_base_url == "https://api.fireworks.ai/inference/v1"
        assert len(settings.allowed_models) == 5

    def test_roles_assigned_correctly(self, settings):
        assert "minimax-m3" in settings.roles.default
        assert "minimax-m3" in settings.roles.reasoning
        assert "kimi-k2p7-code" in settings.roles.code

    def test_gemma_models_detected(self, settings):
        assert len(settings.roles.gemma_models) == 3
        assert all("gemma" in m for m in settings.roles.gemma_models)

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="FIREWORKS_API_KEY"):
            load_settings()

    def test_missing_base_url_raises(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
        with pytest.raises(ConfigError, match="FIREWORKS_BASE_URL"):
            load_settings()

    def test_missing_allowed_models_raises(self, monkeypatch):
        monkeypatch.delenv("ALLOWED_MODELS", raising=False)
        with pytest.raises(ConfigError, match="ALLOWED_MODELS"):
            load_settings()


class TestIsModelAllowed:
    """Settings.is_model_allowed() guard'inin dogru calistigini dogrula."""

    def test_allowed_model(self, settings):
        assert settings.is_model_allowed("accounts/fireworks/models/minimax-m3") is True

    def test_disallowed_model(self, settings):
        assert settings.is_model_allowed("accounts/fireworks/models/gpt-4") is False

    def test_gemma_deployment_scoped(self, settings):
        """Deployment-scoped Gemma model ID'si (# suffix) kabul edilmeli."""
        assert settings.is_model_allowed(
            "accounts/fireworks/models/gemma-4-26b-a4b-it#my-acct/my-deploy"
        ) is True


class TestGemmaSettings:
    """Gemma deployment ayarlarinin yuklenmes."""

    def test_gemma_configured(self, settings):
        assert settings.gemma.is_configured is True
        assert "gemma" in settings.gemma.model

    def test_gemma_not_ready_to_query(self, settings):
        """GEMMA_QUERY_MODEL_ID set edilmedigi icin hazir olmamali."""
        assert settings.gemma.is_ready_to_query is False

    def test_gemma_defaults(self, settings):
        assert settings.gemma.deployment_id == "optiroute-gemma"
        assert settings.gemma.min_replica_count == 0
        assert settings.gemma.max_replica_count == 1

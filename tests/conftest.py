"""
conftest.py
===========
Tum test dosyalari icin ortak pytest fixture'lari.

En onemlisi: gercek Fireworks API'sine ASLA cagri yapilmaz - tum testler
ya saf fonksiyonlari (deterministic_solver, triage heuristic, prompt_compressor,
validator) dogrudan test eder, ya da FireworksClient.chat_completion'i
unittest.mock ile sahteler (bkz. fake_client fixture).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Testler her zaman ayni, bilinen ALLOWED_MODELS setiyle calissin diye
# gercek .env yerine sabit test degerleri kullaniyoruz.
_TEST_ENV = {
    "FIREWORKS_API_KEY": "fw_test_key",
    "FIREWORKS_BASE_URL": "https://api.fireworks.ai/inference/v1",
    "ALLOWED_MODELS": (
        "accounts/fireworks/models/minimax-m3,"
        "accounts/fireworks/models/kimi-k2p7-code,"
        "accounts/fireworks/models/gemma-4-31b-it,"
        "accounts/fireworks/models/gemma-4-26b-a4b-it,"
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
    ),
    "GEMMA_MODEL": "accounts/fireworks/models/gemma-4-26b-a4b-it",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Her testten once ilgili env degiskenlerini bilinen test degerlerine
    sabitler - kullanicinin gercek .env dosyasindan sizinti/etkilesim olmasin."""
    for key, value in _TEST_ENV.items():
        monkeypatch.setenv(key, value)
    yield


@pytest.fixture
def settings():
    from config import load_settings
    return load_settings()


@pytest.fixture
def client(settings):
    from fireworks_client import FireworksClient
    return FireworksClient(settings)


@pytest.fixture
def router(client, settings):
    from router import OptiRouter
    return OptiRouter(client, settings)


def make_completion_result(text: str, model: str = "accounts/fireworks/models/minimax-m3",
                            total_tokens: int = 10, finish_reason: str = "stop"):
    """Testlerde CompletionResult uretmek icin kisa yol."""
    from fireworks_client import CompletionResult
    return CompletionResult(
        text=text,
        model=model,
        prompt_tokens=max(total_tokens - 5, 0),
        completion_tokens=min(total_tokens, 5),
        total_tokens=total_tokens,
        latency_seconds=0.1,
        raw_finish_reason=finish_reason,
    )

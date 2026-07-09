"""
router.py entegrasyon testleri.

Gercek Fireworks API'sine HICBIR cagri yapilmaz - FireworksClient.chat_completion
her testte unittest.mock ile sahtelenir. Amac: OptiRouter.solve()'un tum
orkestrasyon mantigini (Tier 0 -> triage -> Gemma-once-dene -> validate ->
duzeltici retry) uctan uca, gercek modul entegrasyonuyla dogrulamak.
"""

from unittest.mock import patch

from tests.conftest import make_completion_result
from router import RouteSource


def _is_classifier_call(messages):
    return "task-category classifier" in messages[0]["content"]


def test_tier0_short_circuits_before_any_api_call(router, client):
    """Deterministik cozulebilen bir gorev icin HICBIR Fireworks cagrisi
    yapilmamali - 0 token, 0 API call."""
    with patch.object(client, "chat_completion") as mock_call:
        result = router.solve("What is 47 * 89?")

    assert result.text == "4183"
    assert result.source == RouteSource.DETERMINISTIC
    assert result.tokens_spent == 0
    mock_call.assert_not_called()


def test_gemma_success_uses_single_call(router, client):
    """Gemma basarili oldugunda IKINCI (fallback) bir cagri YAPILMAMALI -
    gereksiz token harcanmamali."""
    answer_call_count = {"n": 0}

    def fake(model, messages, **kwargs):
        if _is_classifier_call(messages):
            return make_completion_result("1", total_tokens=41)
        answer_call_count["n"] += 1
        return make_completion_result("Paris is the capital of France.", model=model, total_tokens=30)

    with patch.object(client, "chat_completion", side_effect=fake):
        result = router.solve("What is the capital of France?")

    assert result.source == RouteSource.GEMMA_BONUS
    assert "gemma" in result.model_used
    # Sadece BIR cevap cagrisi olmali (Gemma basarili oldu, fallback'e gerek yok) -
    # triage cagrisi (minimax-m3'e giden siniflandirma cagrisi) bu sayaca DAHIL DEGIL.
    assert answer_call_count["n"] == 1


def test_gemma_failure_falls_back_silently(router, client):
    """Gemma basarisiz olursa exception disari SIZMAMALI, sessizce
    guvenilir serverless modele dusulmeli."""

    def fake(model, messages, **kwargs):
        if _is_classifier_call(messages):
            return make_completion_result("1", total_tokens=41)
        if "gemma" in model:
            raise RuntimeError("deployment not ready (simulated)")
        return make_completion_result("Paris is the capital of France.", model=model, total_tokens=30)

    with patch.object(client, "chat_completion", side_effect=fake):
        result = router.solve("What is the capital of France?")

    assert result.source == RouteSource.DEFAULT_MODEL
    assert "minimax" in result.model_used
    assert result.tokens_spent == 41 + 30


def test_invalid_answer_triggers_corrective_retry(router, client):
    """NER icin gecersiz (JSON olmayan) bir ilk cevap, duzeltici bir retry
    tetiklemeli ve nihai cevap DUZELTILMIS versiyon olmali."""
    prompt = "Extract all named entities from this sentence as JSON: Maria works at Acme."

    def fake(model, messages, **kwargs):
        if len(messages) > 2:  # duzeltici retry (onceki cevap + duzeltme talebi icerir)
            return make_completion_result('[{"text": "Maria", "type": "person"}]', model=model, total_tokens=75)
        return make_completion_result("Maria is a person mentioned in the text.", model=model, total_tokens=40)

    with patch.object(client, "chat_completion", side_effect=fake):
        result = router.solve(prompt)

    assert result.was_corrected is True
    assert result.text == '[{"text": "Maria", "type": "person"}]'
    assert result.tokens_spent == 40 + 75  # bu prompt heuristic ile 0-token siniflandirilir


def test_corrective_retry_failure_keeps_first_answer(router, client):
    """Duzeltici retry de basarisiz olursa (exception), CRASH OLMAMALI -
    ilk (gecersiz olsa da) cevap korunmali."""
    prompt = "Extract all named entities from this sentence as JSON: Maria works at Acme."

    def fake(model, messages, **kwargs):
        if len(messages) > 2:
            raise RuntimeError("network dropped during retry (simulated)")
        return make_completion_result("not valid json at all", model=model, total_tokens=40)

    with patch.object(client, "chat_completion", side_effect=fake):
        result = router.solve(prompt)

    assert result.was_corrected is False
    assert result.text == "not valid json at all"
    assert result.tokens_spent == 40


def test_invalid_gemma_answer_retries_on_default_model(router, client):
    """Gemma'dan gelen geçersiz (JSON olmayan) cevap, default modele (minimax-m3)
    giden düzeltici bir retry tetiklemelidir."""
    prompt = "Extract all named entities from this sentence as JSON: Maria works at Acme."

    call_log = []

    def fake(model, messages, **kwargs):
        call_log.append((model, len(messages)))
        if "gemma" in model:
            # Gemma geçersiz bir ilk cevap dönsün
            return make_completion_result("Gemma returned plain text instead of JSON.", model=model, total_tokens=30)
        # minimax-m3 (default model) düzeltilmiş cevabı dönsün
        return make_completion_result('[{"text": "Maria", "type": "person"}]', model=model, total_tokens=50)

    with patch.object(client, "chat_completion", side_effect=fake):
        result = router.solve(prompt)

    # 1. İlk çağrı Gemma'ya gitmeli (messages uzunluğu <= 2)
    # 2. İkinci çağrı (retry) minimax-m3'e (default model) gitmeli (messages uzunluğu > 2)
    assert call_log[0] == (router._settings.gemma.model, 2)
    assert "minimax" in call_log[1][0]
    assert call_log[1][1] > 2  # retry context'i olmalı
    assert result.was_corrected is True
    assert result.text == '[{"text": "Maria", "type": "person"}]'
    assert result.model_used == "accounts/fireworks/models/minimax-m3"



def test_reasoning_and_code_roles_never_attempt_gemma(router, client):
    """Gemma denemesi SADECE 'default' rolundeki kategoriler icin yapilmali -
    math/logic/code gorevlerinde Gemma'ya hic dokunulmamali."""
    call_log = []

    def fake(model, messages, **kwargs):
        call_log.append(model)
        return make_completion_result("def f(): return 42", model=model, total_tokens=50)

    with patch.object(client, "chat_completion", side_effect=fake):
        router.solve("Write a function that returns 42.")

    assert not any("gemma" in m for m in call_log), "Code gorevinde Gemma'ya YANLISLIKLA cagri yapildi!"

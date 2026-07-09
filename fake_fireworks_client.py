"""
fake_fireworks_client.py
========================
SADECE YEREL GELISTIRME/TEST ICIN. Fireworks kredisi/API key'i olmadan
main.py'yi (ve dolayisiyla tum router.py orkestrasyonunu: Tier 0 -> triage
-> Gemma-once-dene -> validate -> duzeltici retry -> batch I/O -> deadline
yonetimi) UCTAN UCA, gercek network cagrisi OLMADAN calistirabilmek icin.

FireworksClient ile AYNI genel arayuzu (chat_completion(model, messages, **kw)
-> CompletionResult, .usage) tasir, boylece router.py/main.py hicbir
degisiklik gerektirmeden bu sinifi FireworksClient yerine kullanabilir.

ONEMLI: Bu sinif SKORLANAN kosuda KULLANILMAMALI - ciktilari gercek bir
modelden gelmiyor, sadece kaba/kalip-tabanli sahte cevaplar uretiyor. Amaci
tek: "kredi/API key olmadan da pipeline'in COKMEDIGINI ve mantiginin dogru
calistigini gorebilmek". Gercek cevap KALITESI icin README'deki "Yerel
Uctan Uca Test (gercek API ile)" adimini calistirman gerekiyor.

Kontrollu senaryo testi icin ortam degiskenleri:
  FAKE_GEMMA_AVAILABLE=1   -> Gemma denemeleri BASARILI gibi davranilir
                              (varsayilan: basarisiz - gercek dunyada da
                              Gemma deployment'i genelde hazir olmuyor,
                              bu yuzden fallback yolunu varsayilan test eder)
  FAKE_FORCE_INVALID=1     -> ilk cevaplari kasten "supheli" uretir, boylece
                              validator.py + duzeltici retry yolunu da
                              yerel olarak gozlemleyebilirsin
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from config import Settings
from fireworks_client import CompletionResult, DisallowedModelError, UsageTracker


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _fake_token_count(text: str) -> int:
    # Kaba bir yaklasik: ~4 karakter = 1 token. Sadece raporun anlamli
    # gorunmesi icin - gercek tokenizer'a asla esdeger degildir.
    return max(1, len(text) // 4)


_ENTITY_WORD = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def _fake_ner_answer(prompt: str) -> str:
    """Prompt'taki buyuk harfle baslayan kelimelerden kaba bir sahte JSON
    varlik listesi uretir - gercek NER kalitesi DEGIL, sadece 'gecerli JSON
    doner mi' pipeline'ini test etmek icin."""
    words = list(dict.fromkeys(_ENTITY_WORD.findall(prompt)))[:5]
    entities = [{"text": w, "type": "unknown"} for w in words]
    return json.dumps(entities)


def _fake_code_answer(prompt: str) -> str:
    if "palindrome" in prompt.lower():
        return (
            "def is_palindrome(s: str) -> bool:\n"
            "    cleaned = s.replace(' ', '').lower()\n"
            "    return cleaned == cleaned[::-1]"
        )
    return "def solution():\n    # SAHTE (fake) cevap - gercek model degil\n    return 42"


def _fake_generic_answer(prompt: str) -> str:
    return f"[FAKE ANSWER - offline test modu] Bu, '{prompt[:60]}...' istegine sahte bir cevaptir."


def _looks_invalid_on_purpose() -> bool:
    return _env_flag("FAKE_FORCE_INVALID")


@dataclass
class FakeFireworksClient:
    """FireworksClient ile ayni disarisi-gorunen arayuzu tasir ama HICBIR
    gercek network cagrisi yapmaz."""

    _settings: Settings
    _usage: UsageTracker = field(default_factory=UsageTracker)

    def __init__(self, settings: Settings, usage_tracker: Optional[UsageTracker] = None):
        self._settings = settings
        self._usage = usage_tracker or UsageTracker()

    @property
    def usage(self) -> UsageTracker:
        return self._usage

    def chat_completion(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        timeout_seconds: Optional[float] = None,
        retryable: bool = True,
    ) -> CompletionResult:
        if not self._settings.is_model_allowed(model):
            raise DisallowedModelError(
                f"'{model}' ALLOWED_MODELS listesinde degil (fake client bile bu guard'i uyguluyor)."
            )

        # Gemma'ya (deployment gerektiren, on-demand model) yapilan denemeler,
        # varsayilan olarak GERCEK DUNYADAKI EN OLASI durumu simule eder:
        # deployment hazir degil -> exception -> router sessizce fallback yapar.
        if "gemma" in model.lower() and not _env_flag("FAKE_GEMMA_AVAILABLE"):
            raise RuntimeError(
                "[FAKE] Gemma on-demand deployment hazir degil (simule edildi). "
                "Gercekmis gibi denemek icin FAKE_GEMMA_AVAILABLE=1 ayarla."
            )

        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg["content"]  # sistemin son user mesaji yeterli
        system_content = messages[0]["content"] if messages else ""

        # Triage'in model-fallback siniflandirma cagrisini ozel olarak yakala.
        if "task-category classifier" in system_content:
            text = "1"  # varsayilan: factual_knowledge
        elif _looks_invalid_on_purpose() and len(messages) <= 2:
            # Duzeltici retry yolunu yerel test etmek icin BILEREK gecersiz
            # bir ilk cevap uret (retry mesajlarinda messages uzunlugu > 2 olur).
            text = "hmm, not sure, here is some prose without the requested format."
        elif "valid JSON" in system_content or "named entit" in user_content.lower():
            text = _fake_ner_answer(user_content)
        elif "code" in system_content.lower() or "function" in user_content.lower() or "bug" in user_content.lower():
            text = _fake_code_answer(user_content)
        else:
            text = _fake_generic_answer(user_content)

        latency = 0.05  # sahte, aninda "cevap"
        time.sleep(0)  # gercek bir gecikme yok - offline calisiyoruz

        prompt_tokens = _fake_token_count(system_content + user_content)
        completion_tokens = _fake_token_count(text)
        result = CompletionResult(
            text=text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_seconds=latency,
            raw_finish_reason="stop",
        )
        self._usage.record(result)
        return result

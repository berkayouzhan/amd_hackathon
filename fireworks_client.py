"""
fireworks_client.py
====================
Fireworks AI Inference API (OpenAI-uyumlu) etrafinda ince bir wrapper.

Bu modul sistemin TEK skorlanan cikis kapisidir: yarismada token sayimi
buradan gecen her cagriya gore yapiliyor. Bu yuzden:

  1. ALLOWED_MODELS disina TEK BIR cagri bile gitmemeli (config.Settings
     tarafindan guard'lanir).
  2. Her cagrinin token kullanimi (prompt/completion/total) mutlaka
     kaydedilmeli - hem debug icin hem de kendi local "token butcesi"
     takibimiz icin (Adim 2-3'te triage/validation bunu kullanacak).
  3. Gecici ag hatalarinda (timeout, 429, 5xx) retry olmali - aksi halde
     otonom kosan bir agent tek bir gecici hatada tum gorevi kaybeder.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings

logger = logging.getLogger("optiroute.fireworks_client")


class DisallowedModelError(RuntimeError):
    """ALLOWED_MODELS disinda bir modele cagri yapilmaya calisildiginda firlatilir.
    Bu SESSIZCE gecilmemeli - yarisma kurallarina aykiri bir cagri, tum
    submission'i gecersiz kilabilir."""


@dataclass
class CompletionResult:
    """Tek bir Fireworks cagrisinin sonucu - hem cevabi hem de skorlama
    icin kritik olan token/latency bilgisini tasir."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    raw_finish_reason: Optional[str] = None


@dataclass
class UsageTracker:
    """Sürec boyunca kümülatif token kullanımını izler.

    Yarışma skorunun ana bileşeni 'harcanan token' olduğu için, agent'ın
    kendi çalışması sırasında bunu görebilmesi (ör. bir görev için heavy
    modele geçmeden önce 'bu göreve şimdiye kadar ne kadar harcadık' diye
    sorabilmesi) ileride bütçe-farkında (budget-aware) kararlar almamızı
    sağlayacak temel.

    Thread-safe: paralel gorev islemede birden fazla thread ayni anda
    record() cagirabilir."""

    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls_by_model: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, result: CompletionResult) -> None:
        with self._lock:
            self.call_count += 1
            self.prompt_tokens += result.prompt_tokens
            self.completion_tokens += result.completion_tokens
            self.total_tokens += result.total_tokens
            self.calls_by_model[result.model] = self.calls_by_model.get(result.model, 0) + 1

    def report(self) -> str:
        with self._lock:
            lines = [
                f"Toplam cagri        : {self.call_count}",
                f"Toplam prompt token  : {self.prompt_tokens}",
                f"Toplam completion tok: {self.completion_tokens}",
                f"Toplam token (score) : {self.total_tokens}",
                f"Model basina cagri   : {self.calls_by_model}",
            ]
            return "\n".join(lines)


_RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)


class FireworksClient:
    """Fireworks AI'ye giden TUM cagrilarin gectigi tek nokta.

    router.py, triage.py, validators.py gibi ust katmanlar dogrudan
    OpenAI SDK'sini import etmemeli - hepsi bu sinif uzerinden gecmeli.
    Boylece token takibi ve ALLOWED_MODELS guard'i tek bir yerde kalir.
    """

    def __init__(self, settings: Settings, usage_tracker: Optional[UsageTracker] = None):
        self._settings = settings
        self._usage = usage_tracker or UsageTracker()
        self._client = OpenAI(
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
            timeout=settings.request_timeout_seconds,
        )

    @property
    def usage(self) -> UsageTracker:
        return self._usage

    @retry(
        # Guide (genel kural): "Response time per request must be under 30
        # seconds". timeout=20s + 1 retry + max 3s bekleme = kotu senaryoda
        # ~43s olabilir ama bu SADECE Fireworks tarafinda gercek bir sorun
        # varken olur (ilk deneme zaten timeout'a ugramis demektir). Normal
        # basarili cagrida gecikme sadece modelin gercek uretim suresidir.
        # Eskiden 3 deneme + 8s max bekleme (~70s+ potansiyel) kullaniyorduk -
        # bu, 30s kuralini acikca ihlal edebilirdi.
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=3),
        stop=stop_after_attempt(2),
        reraise=True,
    )
    def _call_api(self, **kwargs):
        return self._client.chat.completions.create(**kwargs)

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
        """Fireworks'e tek bir chat completion cagrisi yapar.

        temperature default'u 0.0: token-verimli, deterministik bir router
        icin coğu zaman tutarlilik dogruluktan daha degerlidir - ayni gorev
        tekrar geldiginde ayni (dogru) cevabi almak istiyoruz.

        timeout_seconds / retryable: normalde client-level timeout + retry
        politikasi kullanilir. Ama "once Gemma'yi dene, olmazsa serverless'e
        dus" gibi SPEKULATIF cagrilarda (bkz. router.py), Gemma'nin cevap
        vermemesi/yavas olmasi normal fallback cagrisini da geciktirmemeli -
        bu yuzden boyle cagrilar timeout_seconds=kisa deger + retryable=False
        ile yapilir (tek deneme, hizli-basarisiz-ol).
        """
        if not self._settings.is_model_allowed(model):
            raise DisallowedModelError(
                f"'{model}' ALLOWED_MODELS listesinde degil: {self._settings.allowed_models}. "
                "Bu cagri Fireworks'e GONDERILMEDI (skoru bozmamak icin engellendi)."
            )

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.default_max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if timeout_seconds is not None:
            payload["timeout"] = timeout_seconds

        start = time.monotonic()
        try:
            if retryable:
                response = self._call_api(**payload)
            else:
                response = self._client.chat.completions.create(**payload)
        except APIStatusError as e:
            logger.error("Fireworks API hata dondu (model=%s): %s", model, e)
            raise
        latency = time.monotonic() - start

        choice = response.choices[0]
        usage = response.usage

        result = CompletionResult(
            text=choice.message.content or "",
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_seconds=latency,
            raw_finish_reason=choice.finish_reason,
        )
        self._usage.record(result)
        logger.info(
            "Fireworks cagrisi: model=%s tokens=%d latency=%.2fs",
            model,
            result.total_tokens,
            latency,
        )
        return result

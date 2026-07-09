"""
local_model.py
==============
Konteyner icindeki yerel model (Qwen 2.5 1.5B Instruct GGUF) cikarim modulu.

Sorumluluklari:
  1. Llama-cpp-python kutuphanesini ve model dosyasini yukle (hata durumunda sessizce devredisi kalir).
  2. Qwen chat template ile prompt bicimlendir.
  3. Yerel cikarim yapip sonucu CompletionResult olarak doner.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fireworks_client import CompletionResult

logger = logging.getLogger("optiroute.local_model")

# Konteyner icindeki model dosya yolu (varsayilan)
DEFAULT_LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/app/qwen2.5-1.5b-instruct-q4_k_m.gguf")

_local_llm = None
_initialization_attempted = False


def get_local_llm():
    """Llama modelini bir kez yukler ve hafizada tutar (lazy initialization)."""
    global _local_llm, _initialization_attempted
    if _initialization_attempted:
        return _local_llm

    _initialization_attempted = True

    if not os.path.exists(DEFAULT_LOCAL_MODEL_PATH):
        logger.info("Yerel model dosyasi bulunamadi: %s. Yerel cikarim devre disi.", DEFAULT_LOCAL_MODEL_PATH)
        return None

    try:
        from llama_cpp import Llama

        logger.info("Yerel model yukleniyor: %s...", DEFAULT_LOCAL_MODEL_PATH)
        start = time.monotonic()
        # n_ctx=2048 prompt ve cevap icin yeterli, n_threads varsayilan 4
        _local_llm = Llama(
            model_path=DEFAULT_LOCAL_MODEL_PATH,
            verbose=False,
            n_ctx=2048,
            n_threads=int(os.getenv("LOCAL_MODEL_THREADS", "4")),
        )
        logger.info("Yerel model %.2fs icinde basariyla yuklendi.", time.monotonic() - start)
    except ImportError:
        logger.info("llama-cpp-python yuklu degil. Yerel cikarim devre disi.")
    except Exception as e:
        logger.error("Yerel model yuklenirken hata olustu: %s", e)

    return _local_llm


def run_local_inference(
    system_prompt: str, prompt: str, max_tokens: int
) -> Optional[CompletionResult]:
    """Yerel model ile cikarim yapar. Hata olusursa veya model yuklu degilse None doner."""
    llm = get_local_llm()
    if llm is None:
        return None

    # Qwen chat template bicimlendirmesi
    formatted_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    logger.info("Yerel model ile cikarim baslatiliyor (max_tokens=%d)...", max_tokens)
    start = time.monotonic()
    try:
        response = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministik ciktilar icin
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        latency = time.monotonic() - start
        text = response["choices"][0]["text"].strip()
        finish_reason = response["choices"][0]["finish_reason"] or "stop"

        logger.info("Yerel cikarim %.2fs icinde tamamlandi. Sonlanma nedeni: %s", latency, finish_reason)
        return CompletionResult(
            text=text,
            model="local/qwen2.5-1.5b-instruct",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,  # Liderlik tablosu icin 0 token
            latency_seconds=latency,
            raw_finish_reason=finish_reason,
        )
    except Exception as e:
        logger.error("Yerel cikarim sirasinde hata olustu: %s", e)
        return None

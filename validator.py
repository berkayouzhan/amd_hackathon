"""
validator.py
============
Speculative validation: 0 token harcayarak bir model cevabinin supheli
olup olmadigini kontrol eder (bos/kesilmis/ret/gecersiz-JSON/sayi-yok/kod-yok).
Supheliyse router.py TEK SEFERLIK duzeltici bir retry tetikler.

Guclendirilmis kontroller:
  - Sentiment cevabi pozitif/negatif/neutral icermeli
  - Factual cevap cok kisa (< 10 karakter) olmamali
  - Logic cevap sonuc-bildiren kelime icermeli
  - NER cevabindaki markdown fence'ler strip edilip JSON olarak kontrol edilmeli
  - Summarization: orijinal metnin kopyasi olmamali (TODO: prompt gerekir)
"""

from __future__ import annotations

import json
import re

from triage import TaskCategory

_REFUSAL_PATTERNS = re.compile(
    r"^\s*(i'?m sorry|i cannot|i can't|as an ai\b|i am unable to|i am not able to)",
    re.IGNORECASE,
)

_CODE_SIGNAL = re.compile(
    r"```|(^|\n)\s*def\s+\w+\s*\(|\breturn\b|\bfunction\b|\bclass\s+\w+|"
    r"\bimport\s+\w+|\bfor\s+\w+\s+in\b|\bwhile\s+\w+|\bif\s+\w+",
    re.IGNORECASE,
)

_SENTIMENT_KEYWORDS = re.compile(
    r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE
)

_LOGIC_CONCLUSION = re.compile(
    r"\b(therefore|thus|hence|conclusion|answer:|the answer is|so,?\s+\w+\s+(is|are|owns?|lives?))\b",
    re.IGNORECASE,
)

# Markdown JSON fence stripping icin
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_PATTERNS.match(text.strip()))


def _strip_json_fence(text: str) -> str:
    """NER cevabindaki ```json ... ``` fence'ini strip eder."""
    match = _JSON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _is_valid_json(text: str) -> bool:
    # Once fence strip et, sonra kontrol et
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
        # Gecerli JSON olmali VE list veya dict olmali
        return isinstance(parsed, (list, dict))
    except (json.JSONDecodeError, ValueError):
        return False


def _looks_like_code(text: str) -> bool:
    return bool(_CODE_SIGNAL.search(text))


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _has_sentiment_keyword(text: str) -> bool:
    return bool(_SENTIMENT_KEYWORDS.search(text))


def _has_logic_conclusion(text: str) -> bool:
    return bool(_LOGIC_CONCLUSION.search(text))


def validate(result, category: TaskCategory) -> bool:
    """result: fireworks_client.CompletionResult veya uyumlu nesne.
    True = gecerli, False = supheli (retry tetiklenmeli)."""
    text = (result.text or "").strip()

    # --- Genel kontroller (tum kategoriler) ---
    if not text:
        return False
    if hasattr(result, 'raw_finish_reason') and result.raw_finish_reason == "length":
        return False  # kesilmis cevap
    if _looks_like_refusal(text):
        return False

    # --- Kategori-ozel kontroller ---
    if category == TaskCategory.NAMED_ENTITY_RECOGNITION:
        return _is_valid_json(text)

    if category == TaskCategory.MATHEMATICAL_REASONING:
        return _has_digit(text)

    if category in (TaskCategory.CODE_DEBUGGING, TaskCategory.CODE_GENERATION):
        return _looks_like_code(text)

    if category == TaskCategory.SENTIMENT_CLASSIFICATION:
        return _has_sentiment_keyword(text)

    if category == TaskCategory.LOGICAL_REASONING:
        # Cevap en az 20 karakter olmali (cok kisa = muhtemelen yanlis)
        # VE sonuc-bildiren bir ifade icermeli
        if len(text) < 20:
            return False
        return _has_logic_conclusion(text)

    if category == TaskCategory.FACTUAL_KNOWLEDGE:
        # Cok kisa cevaplar supheli (< 10 karakter)
        if len(text) < 10:
            return False

    if category == TaskCategory.TEXT_SUMMARIZATION:
        # Cok kisa cevaplar supheli (< 15 karakter)
        if len(text) < 15:
            return False

    return True

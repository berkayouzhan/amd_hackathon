"""
validator.py
============
Speculative validation: 0 token harcayarak bir model cevabinin supheli
olup olmadigini kontrol eder (bos/kesilmis/ret/gecersiz-JSON/sayi-yok/kod-yok).
Supheliyse router.py TEK SEFERLIK duzeltici bir retry tetikler.
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
    r"```|(^|\n)\s*def\s+\w+\s*\(|\breturn\b|\bfunction\b",
    re.IGNORECASE,
)


def _looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL_PATTERNS.match(text.strip()))


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _looks_like_code(text: str) -> bool:
    return bool(_CODE_SIGNAL.search(text))


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def validate(result, category: TaskCategory) -> bool:
    """result: fireworks_client.CompletionResult. True = gecerli, False = supheli
    (retry tetiklenmeli)."""
    text = (result.text or "").strip()

    if not text:
        return False
    if result.raw_finish_reason == "length":
        return False  # kesilmis cevap
    if _looks_like_refusal(text):
        return False

    if category == TaskCategory.NAMED_ENTITY_RECOGNITION:
        return _is_valid_json(text)
    if category == TaskCategory.MATHEMATICAL_REASONING:
        return _has_digit(text)
    if category in (TaskCategory.CODE_DEBUGGING, TaskCategory.CODE_GENERATION):
        return _looks_like_code(text)

    return True

"""
triage.py
=========
Gorev metnini Track 1'in 8 resmi kategorisinden birine siniflandirir.

Iki katmanli:
  1. 0-token regex heuristic (coğu gorev burada, ucretsiz siniflandirilir).
  2. Heuristic belirsiz kalirsa: ucuz bir model cagrisi (roles.default,
     kisa max_tokens) "task-category classifier" sistem promptuyla tek bir
     rakam dondurur.

CATEGORY_PROFILES: her kategori icin makul bir max_tokens tahmini tutar -
gercek model ciktilari gorulmeden ince ayarlanamaz (bkz. HANDOFF.md §4.6).
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger("optiroute.triage")


class TaskCategory(Enum):
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    MATHEMATICAL_REASONING = "mathematical_reasoning"
    SENTIMENT_CLASSIFICATION = "sentiment_classification"
    TEXT_SUMMARIZATION = "text_summarization"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    CODE_DEBUGGING = "code_debugging"
    LOGICAL_REASONING = "logical_reasoning"
    CODE_GENERATION = "code_generation"


# Model-fallback classifier cevabinin (1-8) esledigi sabit sira. Bu sirayi
# DEGISTIRME - gecmis siniflandirma cagrilariyla tutarliligi bozar.
_CATEGORY_ORDER: list[TaskCategory] = [
    TaskCategory.FACTUAL_KNOWLEDGE,
    TaskCategory.MATHEMATICAL_REASONING,
    TaskCategory.SENTIMENT_CLASSIFICATION,
    TaskCategory.TEXT_SUMMARIZATION,
    TaskCategory.NAMED_ENTITY_RECOGNITION,
    TaskCategory.CODE_DEBUGGING,
    TaskCategory.LOGICAL_REASONING,
    TaskCategory.CODE_GENERATION,
]

# Kategori -> ("default" | "reasoning" | "code") rol eslemesi.
ROLE_BY_CATEGORY: dict[TaskCategory, str] = {
    TaskCategory.FACTUAL_KNOWLEDGE: "default",
    TaskCategory.SENTIMENT_CLASSIFICATION: "default",
    TaskCategory.TEXT_SUMMARIZATION: "default",
    TaskCategory.NAMED_ENTITY_RECOGNITION: "default",
    TaskCategory.MATHEMATICAL_REASONING: "reasoning",
    TaskCategory.LOGICAL_REASONING: "reasoning",
    TaskCategory.CODE_DEBUGGING: "code",
    TaskCategory.CODE_GENERATION: "code",
}

# Kategori-ozel max_tokens tahminleri - gercek cikti gorulmeden makul
# varsayimlarla sabitlendi (bkz. HANDOFF.md §4.6, ince ayar gerekebilir).
CATEGORY_PROFILES: dict[TaskCategory, int] = {
    TaskCategory.FACTUAL_KNOWLEDGE: 256,
    TaskCategory.MATHEMATICAL_REASONING: 512,
    TaskCategory.SENTIMENT_CLASSIFICATION: 128,
    TaskCategory.TEXT_SUMMARIZATION: 256,
    TaskCategory.NAMED_ENTITY_RECOGNITION: 384,
    TaskCategory.CODE_DEBUGGING: 512,
    TaskCategory.LOGICAL_REASONING: 512,
    TaskCategory.CODE_GENERATION: 512,
}

# Siralama ONEMLI: daha spesifik/guvenilir kaliplar once gelir, boylece
# genel bir "what is" sorusu yanlislikla NER/kod/math'e dusmez.
_HEURISTIC_RULES: list[tuple[TaskCategory, re.Pattern]] = [
    (TaskCategory.NAMED_ENTITY_RECOGNITION,
     re.compile(r"named entit(y|ies)", re.IGNORECASE)),
    (TaskCategory.CODE_DEBUGGING,
     re.compile(r"\bbug\b|\bdebug(ging)?\b|\bfix (it|the bug|this function|this code)\b|\bcorrect this code\b|\brefactor\b|\berror in\b|\boptimize\b|\brecursion\b|\bmemoization\b", re.IGNORECASE)),
    (TaskCategory.CODE_GENERATION,
     re.compile(r"\bwrite\s+(a|an|the)\b[^.\n]*\bfunction\b|\bwrite code\b|\bimplement\s+(a|an|the)\b|\bpython script\b|\bwrite\s+(a|an|the)\b[^.\n]*\bprogram\b", re.IGNORECASE)),
    (TaskCategory.SENTIMENT_CLASSIFICATION,
     re.compile(r"\bsentiment\b|\bclassify\s+(the\s+)?tone\b|\btone\s+of\b|\bpositive\s+or\s+negative\b", re.IGNORECASE)),
    (TaskCategory.TEXT_SUMMARIZATION,
     re.compile(r"\bsummar(i[sz]e|y|i[sz]ation)\b", re.IGNORECASE)),
    (TaskCategory.LOGICAL_REASONING,
     re.compile(r"who (owns|is|has)\b|each own[s]? a different|logic puzzle|"
                r"true or false|\bpuzzle\b|\briddle\b|\bdeduce\b", re.IGNORECASE)),
    (TaskCategory.MATHEMATICAL_REASONING,
     re.compile(r"\$\s?\d|\d+\s?%|discount(ed)?|per\s?cent|percentage|\bcalculate\b|\bequation\b|\bcompute\b|\bplus\b|\bminus\b|\bmultiplied\b|\bdivided\b|\bsum\s+of\b|\bfraction\b|\bratio\b", re.IGNORECASE)),
]

_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a task-category classifier. Read the user's task and reply with "
    "ONLY the single digit (1-8) of the best-matching category, nothing else.\n"
    "1. factual_knowledge\n"
    "2. mathematical_reasoning\n"
    "3. sentiment_classification\n"
    "4. text_summarization\n"
    "5. named_entity_recognition\n"
    "6. code_debugging\n"
    "7. logical_reasoning\n"
    "8. code_generation"
)

_DIGIT_PATTERN = re.compile(r"[1-8]")


def _heuristic_classify(prompt: str) -> Optional[TaskCategory]:
    for category, pattern in _HEURISTIC_RULES:
        if pattern.search(prompt):
            return category
    return None


def _parse_classifier_response(text: str) -> TaskCategory:
    match = _DIGIT_PATTERN.search(text or "")
    if not match:
        logger.warning("Siniflandirici model beklenmeyen bir cevap dondu: %r - factual_knowledge'a dusuluyor.", text)
        return TaskCategory.FACTUAL_KNOWLEDGE
    index = int(match.group()) - 1
    return _CATEGORY_ORDER[index]


def classify(prompt: str, client, settings) -> tuple[TaskCategory, int]:
    """Prompt'u siniflandirir. Doner: (kategori, bu siniflandirma icin
    harcanan token - heuristic eslesirse 0)."""
    heuristic_hit = _heuristic_classify(prompt)
    if heuristic_hit is not None:
        return heuristic_hit, 0

    result = client.chat_completion(
        model=settings.roles.default,
        messages=[
            {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0.0,
    )
    category = _parse_classifier_response(result.text)
    return category, result.total_tokens

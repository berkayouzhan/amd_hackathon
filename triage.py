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

# Kategori-ozel max_tokens tahminleri - ince ayar yapildi (v2):
#   - Sentiment: 64 -> 48 (sadece "negative. Bir cumle." yeterli)
#   - Factual: 192 -> 160 (1-3 cumle yeterli)
#   - Math: 384 -> 320 (concise step-by-step + sayi)
#   - Summary: 256 -> 192 (ozet zaten kisa olmali)
#   - NER: 256 -> 192 (JSON listesi genelde kisa)
#   - Code: 768 (degistirilmedi - buyuk fonksiyonlar kesilmesin)
#   - Logic: 512 -> 384 (kisa adimlar + sonuc)
CATEGORY_PROFILES: dict[TaskCategory, int] = {
    TaskCategory.FACTUAL_KNOWLEDGE: 160,
    TaskCategory.MATHEMATICAL_REASONING: 320,
    TaskCategory.SENTIMENT_CLASSIFICATION: 48,
    TaskCategory.TEXT_SUMMARIZATION: 192,
    TaskCategory.NAMED_ENTITY_RECOGNITION: 192,
    TaskCategory.CODE_DEBUGGING: 768,
    TaskCategory.LOGICAL_REASONING: 384,
    TaskCategory.CODE_GENERATION: 768,
}

# Siralama ONEMLI: daha spesifik/guvenilir kaliplar once gelir, boylece
# genel bir "what is" sorusu yanlislikla NER/kod/math'e dusmez.
#
# Guclendirilmis: daha fazla prompt'u 0-token'la siniflandirmak icin
# ek kaliplar eklendi.
_HEURISTIC_RULES: list[tuple[TaskCategory, re.Pattern]] = [
    # --- En spesifik olanlar once ---
    (TaskCategory.NAMED_ENTITY_RECOGNITION,
     re.compile(r"named entit(y|ies)|\bextract\b[^.\n]*\b(entities|names|persons|organizations|locations)\b|\bNER\b|\bas\s+JSON\b", re.IGNORECASE)),
    (TaskCategory.CODE_DEBUGGING,
     re.compile(r"\bbug\b|\bdebug(ging)?\b|\bfix\s+(it|the\s+bug|this\s+function|this\s+code|the\s+code|the\s+error)\b|"
                r"\bcorrect\s+this\s+code\b|\brefactor\b|\berror\s+in\b|\boptimize\b|"
                r"\brecursion\b|\bmemoization\b|\bhas\s+a\s+(bug|problem|issue)\b|"
                r"\bwhat('s|\s+is)\s+wrong\s+with\b|\brace\s+condition\b|\bconcurrency\s+(issue|bug|problem)\b|"
                r"\bcrash(es|ing)?\b|\bfails?\s+(with|when|for)\b|\bnot\s+working\b|"
                r"\bfind\s+the\s+error\b|\bdoesn'?t\s+work\s+as\s+expected\b|"
                r"\bincorrect\s+output\b|\bexpected\s+.*?\bactual\b|"
                r"\b(TypeError|ValueError|IndexError|KeyError|AttributeError|NameError|SyntaxError|RuntimeError)\b", re.IGNORECASE)),
    (TaskCategory.CODE_GENERATION,
     re.compile(r"\bwrite\s+(a|an|the)\b[^.\n]*\b(function|class|method|script|program|code)\b|"
                r"\bwrite\s+code\b|\bimplement\s+(a|an|the)\b|\bpython\s+script\b|"
                r"\bcreate\s+(a|an|the)\b[^.\n]*\b(function|class|method)\b|"
                r"\bcode\s+(that|which|to)\b|\brespond\s+with\s+code\b|"
                r"\bgenerate\s+(a|an|the)?\s*(function|class|code|script)\b|"
                r"\bdevelop\s+(a|an|the)\b|\bbuild\s+(a|an|the)\b[^.\n]*\b(function|class|module|api)\b|"
                r"\bdesign\s+(a|an)\b[^.\n]*\b(function|class|system|module)\b|"
                r"\bprogram\s+that\b|\balgorithm\s+(for|to)\b", re.IGNORECASE)),
    (TaskCategory.SENTIMENT_CLASSIFICATION,
     re.compile(r"\bsentiment\b|\bclassify\s+(the\s+)?(tone|sentiment|feeling)\b|"
                r"\btone\s+of\b|\bpositive\s+or\s+negative\b|"
                r"\bpositive,?\s+negative,?\s+(or|and)\s+neutral\b|"
                r"\bclassify\s+this\s+review\b|\bopinion\s+mining\b|"
                r"\bhow\s+does\s+the\s+(author|reviewer|writer)\s+feel\b", re.IGNORECASE)),
    (TaskCategory.TEXT_SUMMARIZATION,
     re.compile(r"\bsummar(i[sz]e|y|i[sz]ation|i[sz]ing)\b|\bin\s+\d+\s+words\s+or\s+less\b|"
                r"\bbriefly\s+describe\b|\bgive\s+(a|the)\s+gist\b|"
                r"\bcondense\b|\bshorten\b|\bin\s+brief\b|"
                r"\bkey\s+(points?|takeaways?|findings?)\b|\btl;?dr\b|"
                r"\bmain\s+idea\b|\bin\s+one\s+sentence\b|"
                r"\bwhat\s+is\s+this\s+(?:text\s+)?about\b|\bparaphrase\b", re.IGNORECASE)),
    (TaskCategory.LOGICAL_REASONING,
     re.compile(r"who\s+(owns|is|has|lives)\b|each\s+own[s]?\s+a\s+different|"
                r"logic\s+puzzle|true\s+or\s+false|\bpuzzle\b|\briddle\b|"
                r"\bdeduce\b|\bdeduct(ion|ive)\b|\bknights?\s+and\s+knaves?\b|"
                r"\bif.*then.*who\b|\bclue[s]?\b|\bgiven\s+that\b|"
                r"\bsit\s+in\s+a\s+row\b|\bin\s+a\s+row\b[^.]*\bwho\b|"
                r"\bwhat\s+is\s+the\s+order\b|\barrange\b[^.]*\b(order|sequence)\b|"
                r"\bsyllogism\b|\binfer\b|\bconclusion\s+(?:is|follows)\b|"
                r"\ball\s+\w+\s+are\b[^.]*\bsome\b", re.IGNORECASE)),
    (TaskCategory.MATHEMATICAL_REASONING,
     re.compile(r"\$\s?\d|\d+\s?%|discount(ed)?|per\s?cent|percentage|\bcalculate\b|"
                r"\bequation\b|\bcompute\b|\bplus\b|\bminus\b|\bmultiplied\b|"
                r"\bdivided\b|\bsum\s+of\b|\bfraction\b|\bratio\b|"
                r"\bcost[s]?\b|\bprice\b|\btotal\b|\bhow\s+much\b|"
                r"\bhow\s+many\b|\bresult\s+of\b|\bsolve\s+step\s+by\s+step\b|"
                r"\bapples?\b[^.]*\b(sells?|gives?|buys?)\b|"
                r"\binterest\s+rate\b|\bprofit\b|\brevenue\b|\bmargin\b|"
                r"\barea\b|\bvolume\b|\bperimeter\b|\bcircumference\b|"
                r"\baverage\b|\bmean\b|\bmedian\b|"
                r"\bprobability\b|\bcombination\b|\bpermutation\b|\bfactorial\b|"
                r"\bderivative\b|\bintegral\b|\blimit\b|\bgeometr(y|ic)\b", re.IGNORECASE)),
    # --- Factual Knowledge: fallback - soru kaliplari (diger kategorilere esmesmediyse) ---
    (TaskCategory.FACTUAL_KNOWLEDGE,
     re.compile(r"^\s*(what|who|when|where|which|how)\s+(is|was|are|were|did|does|do|many|much)\b|"
                r"\bexplain\b|\bdescribe\b|\bdefine\b|\bwhat\s+year\b|\bname\s+the\b|"
                r"\btell\s+me\s+about\b|\bwhat\s+happens\s+when\b|\bwhy\s+does\b|"
                r"\bwhat\s+causes\b|\bwhat\s+are\s+the\b|"
                r"\blist\s+the\b|\bname\s+(three|five|ten|\d+)\b|\bgive\s+examples?\s+of\b|"
                r"\bwhat\s+is\s+the\s+difference\b|\bcompare\b[^.\n]*\band\b", re.IGNORECASE)),
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

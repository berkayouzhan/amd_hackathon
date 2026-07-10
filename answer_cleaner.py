"""
answer_cleaner.py
==================
Model cevaplarini results.json'a yazmadan once temizler.

Modeller bazen istenmeyen bicimde cevap veriyor:
  - NER icin ```json ... ``` fence'leri
  - Kod icin ```python ... ``` fence'leri
  - "Sure, here's..." gibi intro cumleleri
  - Basta/sonda gereksiz bosluk/newline

Bu modul, kategori-bazli temizleme yapar - 0 token harcar,
sadece string islemleri. LLM-Judge'un bekledigine yakin format
uretmek amaci tasir.
"""

from __future__ import annotations

import re
from typing import Optional

from triage import TaskCategory

# Markdown code fence pattern'leri
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:python|py|javascript|js|java|c|cpp|csharp|go|rust|ruby|typescript|ts)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

# Intro/outro kaliplari - modellerin sik kullandigi gereksiz girisler.
# Sadece giris cumlesi olan ILK SATIRI strip edilir.
# Eger intro'dan sonra yeni satir varsa -> intro satirini kaldir.
# Eger intro'dan sonra ayni satirda asil cevap geliyorsa -> intro'yu kaldir.
_INTRO_LINE = re.compile(
    r"^(?:Sure[,!.]?\s*(?:here(?:'s| is| are))?[^.\n]*[.!]?\s*\n|"
    r"Of course[,!.]?\s*[^.\n]*[.!]?\s*\n|"
    r"Certainly[,!.]?\s*[^.\n]*[.!]?\s*\n|"
    r"Here(?:'s| is| are)\s+[^:\n]*:\s*\n|"
    r"(?:The |My )?(?:answer|solution|result|response)\s*(?:is|:)\s*\n|"
    r"Let me\s+[^.\n]*[.!]?\s*\n|"
    r"I'?d be happy to\s+[^.\n]*[.!]?\s*\n|"
    r"Great question[,!.]?\s*[^.\n]*[.!]?\s*\n|"
    r"Absolutely[,!.]?\s*[^.\n]*[.!]?\s*\n|"
    r"Here you go[,!:]?\s*\n|"
    r"Here's what I (?:found|came up with)[^.\n]*[.!:]?\s*\n)",
    re.IGNORECASE | re.MULTILINE,
)

# Inline intro: "Sure! Paris is..." -> "Paris is..."
_INTRO_INLINE = re.compile(
    r"^(?:Sure[,!.]\s*|Of course[,!.]\s*|Certainly[,!.]\s*|"
    r"Absolutely[,!.]\s*|Great question[,!.]\s*)",
    re.IGNORECASE,
)

# Math: son satirdaki sayiyi cikar
_MATH_FINAL_NUMBER = re.compile(r"^[\s$]*(-?\d[\d,]*\.?\d*)\s*$", re.MULTILINE)

# Sadece NER/JSON cevaplari icin: fence icerigini cikar
def _strip_json_fence(text: str) -> str:
    match = _JSON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text


# Kod cevaplari icin: fence icerigini cikar
def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text


# Genel intro cumlesini kaldir
def _strip_intro(text: str) -> str:
    # Fase 1: Satir bazli intro (intro\ncevap) - yeni satir ile ayrilan intro
    cleaned = _INTRO_LINE.sub("", text, count=1).strip()
    if cleaned and cleaned != text.strip():
        return cleaned

    # Fase 2: Inline intro (Sure! cevap / Of course! cevap)
    cleaned = _INTRO_INLINE.sub("", text, count=1).strip()
    if cleaned and cleaned != text.strip():
        return cleaned

    # Hicbir intro bulunamadiysa orijinali don
    return text


def clean(text: str, category: Optional[TaskCategory] = None) -> str:
    """Kategori-bazli cevap temizleme. 0 token harcar."""
    if not text:
        return text

    text = text.strip()

    if category == TaskCategory.NAMED_ENTITY_RECOGNITION:
        # JSON fence'lerini strip et
        text = _strip_json_fence(text)
        # Bazi modeller JSON'dan once aciklama yaziyor - son [ ile baslayan kismi bul
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                # Bu satirdan itibaren JSON olmali
                remaining = "\n".join(lines[i:]).strip()
                # Gecerli JSON olup olmadigini kontrol et
                try:
                    import json
                    json.loads(remaining)
                    text = remaining
                except (json.JSONDecodeError, ValueError):
                    pass
                break

    elif category in (TaskCategory.CODE_DEBUGGING, TaskCategory.CODE_GENERATION):
        # Kod fence'lerini strip et (fence varsa icerigi al, yoksa oldugu gibi birak)
        if "```" in text:
            text = _strip_code_fence(text)
        # Intro cumlesini kaldir (sadece kod oncesi prose varsa)
        text = _strip_intro(text)

    elif category == TaskCategory.SENTIMENT_CLASSIFICATION:
        # Intro cumlesini kaldir
        text = _strip_intro(text)

    elif category == TaskCategory.MATHEMATICAL_REASONING:
        # Intro cumlesini kaldir
        text = _strip_intro(text)

    elif category == TaskCategory.FACTUAL_KNOWLEDGE:
        # Intro cumlesini kaldir
        text = _strip_intro(text)

    elif category == TaskCategory.TEXT_SUMMARIZATION:
        # Intro cumlesini kaldir
        text = _strip_intro(text)

    elif category == TaskCategory.LOGICAL_REASONING:
        # Intro cumlesini kaldir
        text = _strip_intro(text)

    # Genel: basta/sonda gereksiz newline temizligi
    text = text.strip()

    return text

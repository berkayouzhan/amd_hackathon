"""
prompt_compressor.py
=====================
Gereksiz bosluk/bos satirlari sikistirarak prompt token maliyetini dusurur.

KRITIK KISIT: ```kod blogu``` iceren kisimlar BIREBIR korunur - girinti/bosluk
kod semantigi tasidigi icin (Python girintisi gibi) buraya dokunmak validator'in
"kod-yok" kontrolunu ya da modelin kodu doGRU anlamasini bozabilir.

Ek sikistirmalar (guclendirilmis):
  - Tekrarlayan noktalama (!!!!, ????) tek karaktere indirilir
  - Markdown baslik isaretleri (###) strip edilir (model icin gereksiz)
  - URL'ler kaldirilir (gorev baglaminda genelde gereksiz)
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_TRAILING_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")

# Ek sikistirma kaliplari
_REPEATED_PUNCTUATION = re.compile(r"([!?.]){2,}")
_MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_URL_PATTERN = re.compile(r"https?://\S+")
_REPEATED_DASHES = re.compile(r"-{3,}")


def _compress_plain_text(segment: str) -> str:
    segment = _TRAILING_SPACE_BEFORE_NEWLINE.sub("\n", segment)
    segment = _MULTI_SPACE.sub(" ", segment)
    segment = _MULTI_BLANK_LINE.sub("\n\n", segment)
    # Tekrarlayan noktalama: "!!!" -> "!", "???" -> "?"
    segment = _REPEATED_PUNCTUATION.sub(r"\1", segment)
    # Markdown baslik isaretleri: "### Title" -> "Title"
    segment = _MARKDOWN_HEADER.sub("", segment)
    # URL'leri kaldir (gorev icerigi icin genelde gereksiz)
    segment = _URL_PATTERN.sub("", segment)
    # Tekrarlayan tireler: "---" -> "-"
    segment = _REPEATED_DASHES.sub("-", segment)
    return segment


def compress_prompt(text: str) -> str:
    """Kod bloklarina dokunmadan geri kalan metindeki fazla boslugu sikistirir."""
    if not text:
        return text

    parts = _CODE_FENCE.split(text)
    compressed_parts = [
        part if part.startswith("```") else _compress_plain_text(part)
        for part in parts
    ]
    return "".join(compressed_parts).strip()


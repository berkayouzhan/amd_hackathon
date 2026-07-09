"""
prompt_compressor.py
=====================
Gereksiz bosluk/bos satirlari sikistirarak prompt token maliyetini dusurur.

KRITIK KISIT: ```kod blogu``` iceren kisimlar BIREBIR korunur - girinti/bosluk
kod semantigi tasidigi icin (Python girintisi gibi) buraya dokunmak validator'in
"kod-yok" kontrolunu ya da modelin kodu doGRU anlamasini bozabilir.
"""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE = re.compile(r"\n{3,}")
_TRAILING_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")


def _compress_plain_text(segment: str) -> str:
    segment = _TRAILING_SPACE_BEFORE_NEWLINE.sub("\n", segment)
    segment = _MULTI_SPACE.sub(" ", segment)
    segment = _MULTI_BLANK_LINE.sub("\n\n", segment)
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

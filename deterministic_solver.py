"""
deterministic_solver.py
========================
Tier 0: hicbir Fireworks cagrisi yapmadan (0 token), guvenli/dar bir kalip
setiyle cozulebilen gorevleri yakalar.

BILEREK KONSERVATIF: sadece "cift operandli duz aritmetik" ("What is 47 * 89?"
gibi) yakalanir. Kelime problemleri ("Bir gomlek 40 dolar...") kasitli olarak
BURADA COZULMEZ - yanlis bir tahmin, LLM-Judge accuracy gate'ini dogrudan
tehlikeye atar; oysa bu kategori zaten ucuz bir modele (reasoning rolu)
yonlendirilecek, 0 token kazanci riske degmez.
"""

from __future__ import annotations

import re
from typing import Optional

# "What is 47 * 89?" / "47*89" / "12.5 / 4" / "100 - 37" gibi TEK adimli,
# iki sayili duz aritmetik ifadeleri yakalar. Kelime problemlerinde araya
# giren metin (birim, para, yuzde, "then", "discount" vb.) bu kaliba
# UYMAZ ve None doner - bilerek.
_ARITH_PATTERN = re.compile(
    r"^\s*(?:what\s+is\s+)?(-?\d+(?:\.\d+)?)\s*([*x×/+-])\s*(-?\d+(?:\.\d+)?)\s*\??\s*$",
    re.IGNORECASE,
)


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    # Gereksiz kayan nokta gurultusunu (0.1+0.2 gibi) temizlemek icin
    # makul bir hassasiyette yuvarla, sonra sondaki sifirlari at.
    return f"{value:.10f}".rstrip("0").rstrip(".")


def try_solve(prompt: str) -> Optional[str]:
    """Prompt tek-adimli duz aritmetige uyuyorsa sonucu string olarak doner,
    aksi halde None doner (bu durumda router bir sonraki tier'a - triage'a - gecer)."""
    match = _ARITH_PATTERN.match(prompt.strip())
    if not match:
        return None

    left_raw, op, right_raw = match.groups()
    left, right = float(left_raw), float(right_raw)

    if op in ("*", "x", "×"):
        result = left * right
    elif op == "/":
        if right == 0:
            return None  # sifira bolme - guvenli tarafta kal, LLM'e birak
        result = left / right
    elif op == "+":
        result = left + right
    elif op == "-":
        result = left - right
    else:
        return None

    return _format_number(result)

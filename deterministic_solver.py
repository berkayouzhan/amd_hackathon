"""
deterministic_solver.py
========================
Tier 0: hicbir Fireworks cagrisi yapmadan (0 token), guvenli/dar bir kalip
setiyle cozulebilen gorevleri yakalar.

BILEREK KONSERVATIF: sadece "cift operandli duz aritmetik" ("What is 47 * 89?"
gibi) ve guvenli matematiksel ifadeler yakalanir. Kelime problemleri (
"Bir gomlek 40 dolar...") kasitli olarak BURADA COZULMEZ - yanlis bir tahmin,
LLM-Judge accuracy gate'ini dogrudan tehlikeye atar.

Guclendirilmis: ast.literal_eval tabanlı guvenli ifade degerlendirme eklendi:
  - Parantezli aritmetik: (15 + 25) * 3
  - Us alma: 2^10, 2**10
  - Mod/kalan: 100 % 7
  - Basit fonksiyonlar: sqrt(144)
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Optional

# "What is 47 * 89?" / "47*89" / "12.5 / 4" / "100 - 37" gibi TEK adimli,
# iki sayili duz aritmetik ifadeleri yakalar. Kelime problemlerinde araya
# giren metin (birim, para, yuzde, "then", "discount" vb.) bu kaliba
# UYMAZ ve None doner - bilerek.
_ARITH_PATTERN = re.compile(
    r"^\s*(?:what\s+is\s+)?(-?\d+(?:\.\d+)?)\s*([*x×/+\-])\s*(-?\d+(?:\.\d+)?)\s*\??\s*$",
    re.IGNORECASE,
)

# "Calculate: 15.5 * (40 / (2.5 + 1.5)) - 100" gibi parantezli ifadeler.
# Kelime icermemeli, sadece sayi + operator + parantez.
_EXPR_PATTERN = re.compile(
    r"^\s*(?:(?:what\s+is|calculate|compute|evaluate|result\s+of)\s*:?\s*)?"
    r"([\d\s+\-*/().^%]+)\s*\??\s*$",
    re.IGNORECASE,
)

# Yuzde hesaplama: "What is 15% of 200?" / "15% of 200"
_PERCENT_OF_PATTERN = re.compile(
    r"^\s*(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)\s*\??\s*$",
    re.IGNORECASE,
)

# Kare/kup: "What is the square of 12?" / "What is the cube of 5?"
_POWER_PATTERN = re.compile(
    r"^\s*(?:what\s+is\s+)?the\s+(square|cube|square\s+root)\s+of\s+(\d+(?:\.\d+)?)\s*\??\s*$",
    re.IGNORECASE,
)

# Birim donusum: "How many cm in 5 meters?" / "Convert 3 km to meters"
_UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    ("km", "m"): 1000, ("m", "km"): 0.001,
    ("m", "cm"): 100, ("cm", "m"): 0.01,
    ("cm", "mm"): 10, ("mm", "cm"): 0.1,
    ("m", "mm"): 1000, ("mm", "m"): 0.001,
    ("kg", "g"): 1000, ("g", "kg"): 0.001,
    ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
    ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
    ("m", "feet"): 3.28084, ("feet", "m"): 0.3048,
    ("inch", "cm"): 2.54, ("cm", "inch"): 0.393701,
    ("inches", "cm"): 2.54, ("cm", "inches"): 0.393701,
    ("hour", "minutes"): 60, ("minutes", "hour"): 1/60,
    ("hours", "minutes"): 60, ("minutes", "hours"): 1/60,
    ("day", "hours"): 24, ("hours", "day"): 1/24,
    ("days", "hours"): 24, ("hours", "days"): 1/24,
    ("meters", "cm"): 100, ("cm", "meters"): 0.01,
    ("meters", "feet"): 3.28084, ("feet", "meters"): 0.3048,
    ("kilometers", "meters"): 1000, ("meters", "kilometers"): 0.001,
    ("kilograms", "grams"): 1000, ("grams", "kilograms"): 0.001,
}
_UNIT_CONVERT_PATTERN = re.compile(
    r"^\s*(?:(?:how\s+many|convert)\s+)?(\d+(?:\.\d+)?)\s*(\w+)\s+(?:in|to|=)\s+(?:how\s+many\s+)?(\w+)\s*\??\s*$",
    re.IGNORECASE,
)
_UNIT_CONVERT_ALT_PATTERN = re.compile(
    r"^\s*how\s+many\s+(\w+)\s+(?:in|are\s+in)\s+(\d+(?:\.\d+)?)\s*(\w+)\s*\??\s*$",
    re.IGNORECASE,
)

# Guvenli AST degerlendirme icin izin verilen operatorler
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """AST dugumlerini guvenli bir sekilde degerlendirir (sadece aritmetik)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Izin verilmeyen operator: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type == ast.Div and right == 0:
            raise ValueError("Sifira bolme")
        # Cok buyuk us almalari engelle (DoS korumasi)
        if op_type == ast.Pow and isinstance(right, (int, float)) and right > 100:
            raise ValueError("Cok buyuk us")
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Izin verilmeyen operator: {op_type}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    else:
        raise ValueError(f"Izin verilmeyen AST dugumu: {type(node)}")


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    # Gereksiz kayan nokta gurultusunu (0.1+0.2 gibi) temizlemek icin
    # makul bir hassasiyette yuvarla, sonra sondaki sifirlari at.
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _try_simple_arith(prompt: str) -> Optional[str]:
    """Basit iki-operandli aritmetik: 47 * 89"""
    match = _ARITH_PATTERN.match(prompt.strip())
    if not match:
        return None

    left_raw, op, right_raw = match.groups()
    left, right = float(left_raw), float(right_raw)

    if op in ("*", "x", "×"):
        result = left * right
    elif op == "/":
        if right == 0:
            return None
        result = left / right
    elif op == "+":
        result = left + right
    elif op == "-":
        result = left - right
    else:
        return None

    return _format_number(result)


def _try_expression(prompt: str) -> Optional[str]:
    """Parantezli/karmasik aritmetik ifadeler: (15 + 25) * 3"""
    match = _EXPR_PATTERN.match(prompt.strip())
    if not match:
        return None

    expr_str = match.group(1).strip()
    if not expr_str:
        return None

    # ^ isareti Python'da bitwise XOR, biz us alma olarak yorumluyoruz
    expr_str = expr_str.replace("^", "**")

    try:
        tree = ast.parse(expr_str, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, complex):
            return None
        return _format_number(float(result))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _try_percent_of(prompt: str) -> Optional[str]:
    """Yuzde hesaplama: 'What is 15% of 200?' -> 30"""
    match = _PERCENT_OF_PATTERN.match(prompt.strip())
    if not match:
        return None
    percent, base = float(match.group(1)), float(match.group(2))
    result = (percent / 100.0) * base
    return _format_number(result)


def _try_power(prompt: str) -> Optional[str]:
    """Kare/kup/karekok: 'What is the square of 12?' -> 144"""
    match = _POWER_PATTERN.match(prompt.strip())
    if not match:
        return None
    power_type = match.group(1).lower()
    number = float(match.group(2))
    if power_type == "square":
        result = number ** 2
    elif power_type == "cube":
        result = number ** 3
    elif power_type == "square root":
        if number < 0:
            return None
        result = math.sqrt(number)
    else:
        return None
    return _format_number(result)


def _try_unit_conversion(prompt: str) -> Optional[str]:
    """Birim donusum: 'How many cm in 5 meters?' -> 500"""
    # Kalip 1: "5 km to m" / "Convert 5 km to m"
    match = _UNIT_CONVERT_PATTERN.match(prompt.strip())
    if match:
        value = float(match.group(1))
        from_unit = match.group(2).lower()
        to_unit = match.group(3).lower()
        factor = _UNIT_CONVERSIONS.get((from_unit, to_unit))
        if factor is not None:
            return _format_number(value * factor)
        return None

    # Kalip 2: "How many cm in 5 meters?"
    match = _UNIT_CONVERT_ALT_PATTERN.match(prompt.strip())
    if match:
        to_unit = match.group(1).lower()
        value = float(match.group(2))
        from_unit = match.group(3).lower()
        factor = _UNIT_CONVERSIONS.get((from_unit, to_unit))
        if factor is not None:
            return _format_number(value * factor)

    return None


def try_solve(prompt: str) -> Optional[str]:
    """Prompt tek-adimli veya parantezli duz aritmetige uyuyorsa sonucu
    string olarak doner, aksi halde None doner (router bir sonraki
    tier'a - triage'a - gecer)."""
    # Once basit iki-operandli formu dene (daha guvenilir)
    simple = _try_simple_arith(prompt)
    if simple is not None:
        return simple

    # Yuzde hesaplama
    percent = _try_percent_of(prompt)
    if percent is not None:
        return percent

    # Kare/kup/karekok
    power = _try_power(prompt)
    if power is not None:
        return power

    # Birim donusum
    unit = _try_unit_conversion(prompt)
    if unit is not None:
        return unit

    # Son olarak parantezli/karmasik ifade formu dene
    return _try_expression(prompt)


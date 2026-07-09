"""
test_deterministic_solver.py
=============================
deterministic_solver.try_solve() icin birim testleri.

Kontrol edilenler:
  - Pozitif: tek-adimli aritmetik (+, -, *, /) doğru sonuc verir
  - Negatif: kelime problemleri, coklu islemler, belirsiz ifadeler None doner
  - Kenar durumlari: sifira bolme, ondalikli sayilar, negatif operandlar
"""

from deterministic_solver import try_solve


class TestPositiveCases:
    """try_solve()'un dogru sonuc donmesi gereken durumlar."""

    def test_multiplication_integers(self):
        assert try_solve("What is 47 * 89?") == "4183"

    def test_multiplication_x_operator(self):
        assert try_solve("47 x 89") == "4183"

    def test_addition(self):
        assert try_solve("100 + 37") == "137"

    def test_subtraction(self):
        assert try_solve("100 - 37") == "63"

    def test_division_exact(self):
        assert try_solve("100 / 4") == "25"

    def test_division_decimal(self):
        result = try_solve("10 / 3")
        assert result is not None
        assert float(result) == round(10 / 3, 10)

    def test_decimal_operands(self):
        assert try_solve("12.5 / 4") == "3.125"

    def test_negative_operand(self):
        assert try_solve("-5 + 3") == "-2"

    def test_bare_expression_no_question_mark(self):
        assert try_solve("47*89") == "4183"

    def test_with_leading_trailing_whitespace(self):
        assert try_solve("  What is 6 * 7?  ") == "42"


class TestNegativeCases:
    """try_solve()'un None donmesi gereken durumlar - bunlar LLM'e birakılmalı."""

    def test_word_problem_rejected(self):
        """Kelime problemleri Tier 0'da cozulmemeli."""
        prompt = "A shirt costs $40. It is discounted by 15%. What is the final price?"
        assert try_solve(prompt) is None

    def test_multi_step_rejected(self):
        """Birden fazla islem iceren ifadeler reddedilmeli."""
        assert try_solve("What is 2 + 3 * 4?") is None

    def test_plain_text_rejected(self):
        assert try_solve("What is the capital of France?") is None

    def test_empty_string(self):
        assert try_solve("") is None

    def test_percentage_word_rejected(self):
        assert try_solve("What is 15 percent of 200?") is None


class TestEdgeCases:
    """Kenar durumlari."""

    def test_division_by_zero_returns_none(self):
        """Sifira bolme guvenli tarafta kalarak None donmeli."""
        assert try_solve("10 / 0") is None

    def test_zero_result(self):
        assert try_solve("5 - 5") == "0"

    def test_large_numbers(self):
        result = try_solve("999999 * 999999")
        assert result == "999998000001"

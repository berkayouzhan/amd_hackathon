"""
test_triage.py
===============
triage modulu icin birim testleri.

Kontrol edilenler:
  - Heuristic siniflandirma: 8 kategorinin her biri icin regex eslesmesi
  - Heuristic belirsiz kaldiginda model-fallback cagrisinin tetiklenmesi
  - CATEGORY_PROFILES ve ROLE_BY_CATEGORY'nin eksiksiz olmasi
"""

from unittest.mock import MagicMock

from tests.conftest import make_completion_result
from triage import (
    CATEGORY_PROFILES,
    ROLE_BY_CATEGORY,
    TaskCategory,
    _heuristic_classify,
    classify,
)


class TestHeuristicClassify:
    """0-token regex heuristic'in doğru kategori donmesi gereken durumlar."""

    def test_ner_detection(self):
        assert _heuristic_classify("Extract all named entities from this text.") == TaskCategory.NAMED_ENTITY_RECOGNITION

    def test_code_debugging_bug(self):
        assert _heuristic_classify("Find the bug in this code.") == TaskCategory.CODE_DEBUGGING

    def test_code_debugging_debug(self):
        assert _heuristic_classify("Debug this Python function.") == TaskCategory.CODE_DEBUGGING

    def test_code_debugging_fix(self):
        assert _heuristic_classify("Fix the bug in the following code.") == TaskCategory.CODE_DEBUGGING

    def test_code_generation(self):
        assert _heuristic_classify("Write a function that returns 42.") == TaskCategory.CODE_GENERATION

    def test_code_generation_python(self):
        assert _heuristic_classify("Write a Python function to sort a list.") == TaskCategory.CODE_GENERATION

    def test_sentiment(self):
        assert _heuristic_classify("Classify the sentiment of this review.") == TaskCategory.SENTIMENT_CLASSIFICATION

    def test_summarization(self):
        assert _heuristic_classify("Summarize the following text.") == TaskCategory.TEXT_SUMMARIZATION

    def test_summarization_variant(self):
        assert _heuristic_classify("Provide a summary of this article.") == TaskCategory.TEXT_SUMMARIZATION

    def test_logical_reasoning_puzzle(self):
        assert _heuristic_classify("Solve this logic puzzle: who owns the dog?") == TaskCategory.LOGICAL_REASONING

    def test_logical_reasoning_each_owns(self):
        assert _heuristic_classify("Three friends each own a different pet.") == TaskCategory.LOGICAL_REASONING

    def test_mathematical_discount(self):
        assert _heuristic_classify("A shirt costs $40. It is discounted by 15%.") == TaskCategory.MATHEMATICAL_REASONING

    def test_mathematical_percentage(self):
        assert _heuristic_classify("What is 20 percent of 300?") == TaskCategory.MATHEMATICAL_REASONING

    def test_mathematical_dollar(self):
        assert _heuristic_classify("A ticket costs $50 and there are 3 people.") == TaskCategory.MATHEMATICAL_REASONING

    def test_mathematical_calculate(self):
        assert _heuristic_classify("Calculate the sum of all elements.") == TaskCategory.MATHEMATICAL_REASONING

    def test_mathematical_equation(self):
        assert _heuristic_classify("Solve the following equation.") == TaskCategory.MATHEMATICAL_REASONING

    def test_code_generation_write_code(self):
        assert _heuristic_classify("Write code to reverse a binary tree.") == TaskCategory.CODE_GENERATION

    def test_code_generation_implement(self):
        assert _heuristic_classify("Implement a binary search algorithm.") == TaskCategory.CODE_GENERATION

    def test_code_debugging_correct(self):
        assert _heuristic_classify("Correct this code snippet.") == TaskCategory.CODE_DEBUGGING

    def test_sentiment_tone(self):
        assert _heuristic_classify("Classify the tone of the user feedback.") == TaskCategory.SENTIMENT_CLASSIFICATION

    def test_logic_riddle(self):
        assert _heuristic_classify("Solve this riddle about three boxes.") == TaskCategory.LOGICAL_REASONING

    def test_no_match_returns_none(self):
        """Hicbir regex'e uymayan metin None donmeli (model-fallback tetiklenir)."""
        assert _heuristic_classify("What is the capital of France?") is None


class TestClassifyWithModelFallback:
    """Heuristic eslesmediginde model-fallback siniflandirmanin calismasi."""

    def test_model_fallback_called_when_heuristic_fails(self):
        """'What is the capital of France?' heuristice uymaz, model cagirilmali."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = make_completion_result("1", total_tokens=41)

        mock_settings = MagicMock()
        mock_settings.roles.default = "accounts/fireworks/models/minimax-m3"

        category, tokens = classify("What is the capital of France?", mock_client, mock_settings)

        assert category == TaskCategory.FACTUAL_KNOWLEDGE
        assert tokens == 41
        mock_client.chat_completion.assert_called_once()

    def test_heuristic_match_skips_model_call(self):
        """Heuristic eslesirse model cagirilmamali - 0 token."""
        mock_client = MagicMock()
        mock_settings = MagicMock()

        category, tokens = classify("Classify the sentiment of this text.", mock_client, mock_settings)

        assert category == TaskCategory.SENTIMENT_CLASSIFICATION
        assert tokens == 0
        mock_client.chat_completion.assert_not_called()

    def test_model_returns_invalid_digit_falls_to_factual(self):
        """Model gecersiz bir cevap donerse factual_knowledge'a dusmeli."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = make_completion_result("unknown", total_tokens=30)

        mock_settings = MagicMock()
        mock_settings.roles.default = "accounts/fireworks/models/minimax-m3"

        category, tokens = classify("Some ambiguous task.", mock_client, mock_settings)

        assert category == TaskCategory.FACTUAL_KNOWLEDGE
        assert tokens == 30


class TestCategoryMappingCompleteness:
    """CATEGORY_PROFILES ve ROLE_BY_CATEGORY'nin tum kategorileri kapsamasi."""

    def test_all_categories_have_profiles(self):
        for cat in TaskCategory:
            assert cat in CATEGORY_PROFILES, f"{cat} CATEGORY_PROFILES'ta eksik!"

    def test_all_categories_have_roles(self):
        for cat in TaskCategory:
            assert cat in ROLE_BY_CATEGORY, f"{cat} ROLE_BY_CATEGORY'de eksik!"

    def test_roles_are_valid(self):
        valid_roles = {"default", "reasoning", "code"}
        for cat, role in ROLE_BY_CATEGORY.items():
            assert role in valid_roles, f"{cat} icin gecersiz rol: {role}"

"""
test_answer_cleaner.py
======================
answer_cleaner modulu icin testler.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from answer_cleaner import clean
from triage import TaskCategory


class TestNERCleaning:
    def test_strips_json_fence(self):
        text = '```json\n[{"entity": "Maria", "type": "PERSON"}]\n```'
        result = clean(text, TaskCategory.NAMED_ENTITY_RECOGNITION)
        assert result == '[{"entity": "Maria", "type": "PERSON"}]'

    def test_strips_plain_fence(self):
        text = '```\n[{"entity": "Maria", "type": "PERSON"}]\n```'
        result = clean(text, TaskCategory.NAMED_ENTITY_RECOGNITION)
        assert result == '[{"entity": "Maria", "type": "PERSON"}]'

    def test_strips_prose_before_json(self):
        text = 'Here are the entities:\n[{"entity": "Maria", "type": "PERSON"}]'
        result = clean(text, TaskCategory.NAMED_ENTITY_RECOGNITION)
        assert result == '[{"entity": "Maria", "type": "PERSON"}]'

    def test_empty_array(self):
        text = "```json\n[]\n```"
        result = clean(text, TaskCategory.NAMED_ENTITY_RECOGNITION)
        assert result == "[]"

    def test_no_fence_passthrough(self):
        text = '[{"entity": "Acme", "type": "ORGANIZATION"}]'
        result = clean(text, TaskCategory.NAMED_ENTITY_RECOGNITION)
        assert result == text


class TestCodeCleaning:
    def test_strips_python_fence(self):
        text = '```python\ndef foo():\n    return 42\n```'
        result = clean(text, TaskCategory.CODE_GENERATION)
        assert result == 'def foo():\n    return 42'

    def test_strips_py_fence(self):
        text = '```py\ndef bar():\n    pass\n```'
        result = clean(text, TaskCategory.CODE_GENERATION)
        assert result == 'def bar():\n    pass'

    def test_strips_intro_then_code(self):
        text = "Sure, here's the code:\ndef foo():\n    return 42"
        result = clean(text, TaskCategory.CODE_GENERATION)
        assert "def foo():" in result
        assert "Sure" not in result

    def test_no_fence_code_passthrough(self):
        text = "def hello():\n    print('hello')"
        result = clean(text, TaskCategory.CODE_GENERATION)
        assert result == text

    def test_debugging_strips_fence(self):
        text = '```python\ndef fixed():\n    total += n\n```'
        result = clean(text, TaskCategory.CODE_DEBUGGING)
        assert result == 'def fixed():\n    total += n'


class TestSentimentCleaning:
    def test_strips_intro(self):
        text = "Sure, here's my analysis:\nnegative. The product broke."
        result = clean(text, TaskCategory.SENTIMENT_CLASSIFICATION)
        assert "negative" in result
        assert "Sure" not in result

    def test_no_intro_passthrough(self):
        text = "positive. Great product."
        result = clean(text, TaskCategory.SENTIMENT_CLASSIFICATION)
        assert result == text


class TestFactualCleaning:
    def test_strips_intro(self):
        text = "Of course! The capital of France is Paris."
        result = clean(text, TaskCategory.FACTUAL_KNOWLEDGE)
        assert "Paris" in result
        assert "Of course" not in result


class TestMathCleaning:
    def test_strips_intro(self):
        text = "Certainly! Let me solve this:\nStep 1: 40 * 0.85 = 34\n34"
        result = clean(text, TaskCategory.MATHEMATICAL_REASONING)
        assert "34" in result
        assert "Certainly" not in result


class TestEmptyInput:
    def test_empty_string(self):
        assert clean("", TaskCategory.FACTUAL_KNOWLEDGE) == ""

    def test_none_category(self):
        text = "Some answer"
        assert clean(text, None) == "Some answer"

    def test_whitespace_only(self):
        assert clean("   \n\n  ", TaskCategory.FACTUAL_KNOWLEDGE) == ""


class TestMathAnswerExtraction:
    def test_final_answer_format(self):
        text = "Step 1: 5 + 3 = 8\nStep 2: 8 * 2 = 16\nFinal Answer: 16"
        assert clean(text, TaskCategory.MATHEMATICAL_REASONING) == "16"

    def test_the_answer_is_format(self):
        text = "Some reasoning here\nThe answer is 42.5"
        assert clean(text, TaskCategory.MATHEMATICAL_REASONING) == "42.5"

    def test_answer_with_commas(self):
        text = "Calculation result\nFinal Answer: 1,000,000"
        assert clean(text, TaskCategory.MATHEMATICAL_REASONING) == "1000000"

    def test_no_final_answer_keeps_full_text(self):
        text = "The result of 5 + 3 is clearly 8"
        result = clean(text, TaskCategory.MATHEMATICAL_REASONING)
        assert "8" in result

    def test_general_intro_strip_for_factual(self):
        text = "Sure! Paris is the capital of France."
        result = clean(text, TaskCategory.FACTUAL_KNOWLEDGE)
        assert result == "Paris is the capital of France."


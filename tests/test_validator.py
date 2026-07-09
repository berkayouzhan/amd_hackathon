"""
test_validator.py
==================
validator modulu icin birim testleri.

Kontrol edilenler:
  - Bos/kesilmis/ret cevaplarin reddedilmesi
  - NER icin gecersiz JSON reddedilmesi
  - Math icin rakam yoksa reddedilmesi
  - Code icin kod isareti yoksa reddedilmesi
  - Gecerli cevaplarin onaylanmasi
"""

from tests.conftest import make_completion_result
from triage import TaskCategory
from validator import validate


class TestGeneralChecks:
    """Tum kategoriler icin gecerli olan genel kontroller."""

    def test_empty_text_rejected(self):
        result = make_completion_result("", finish_reason="stop")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is False

    def test_whitespace_only_rejected(self):
        result = make_completion_result("   \n  ", finish_reason="stop")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is False

    def test_truncated_response_rejected(self):
        """finish_reason='length' -> kesilmis cevap -> supheli."""
        result = make_completion_result("Some partial answer...", finish_reason="length")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is False

    def test_refusal_im_sorry(self):
        result = make_completion_result("I'm sorry, I cannot answer that.", finish_reason="stop")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is False

    def test_refusal_as_an_ai(self):
        result = make_completion_result("As an AI, I cannot provide medical advice.", finish_reason="stop")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is False

    def test_refusal_i_cannot(self):
        result = make_completion_result("I cannot assist with that request.", finish_reason="stop")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is False

    def test_valid_factual_answer_accepted(self):
        result = make_completion_result("The capital of France is Paris.", finish_reason="stop")
        assert validate(result, TaskCategory.FACTUAL_KNOWLEDGE) is True

    def test_valid_sentiment_accepted(self):
        result = make_completion_result("The sentiment is negative.", finish_reason="stop")
        assert validate(result, TaskCategory.SENTIMENT_CLASSIFICATION) is True

    def test_valid_summary_accepted(self):
        result = make_completion_result("The council approved a new transit line.", finish_reason="stop")
        assert validate(result, TaskCategory.TEXT_SUMMARIZATION) is True


class TestNERValidation:
    """NER kategorisi icin JSON gecerlilik kontrolu."""

    def test_valid_json_accepted(self):
        result = make_completion_result('[{"text": "Maria", "type": "person"}]', finish_reason="stop")
        assert validate(result, TaskCategory.NAMED_ENTITY_RECOGNITION) is True

    def test_invalid_json_rejected(self):
        result = make_completion_result("Maria is a person mentioned in the text.", finish_reason="stop")
        assert validate(result, TaskCategory.NAMED_ENTITY_RECOGNITION) is False

    def test_empty_json_array_accepted(self):
        result = make_completion_result("[]", finish_reason="stop")
        assert validate(result, TaskCategory.NAMED_ENTITY_RECOGNITION) is True

    def test_json_object_accepted(self):
        result = make_completion_result('{"entities": ["Maria"]}', finish_reason="stop")
        assert validate(result, TaskCategory.NAMED_ENTITY_RECOGNITION) is True


class TestMathValidation:
    """Math kategorisi icin rakam varlik kontrolu."""

    def test_answer_with_number_accepted(self):
        result = make_completion_result("The final price is $30.60.", finish_reason="stop")
        assert validate(result, TaskCategory.MATHEMATICAL_REASONING) is True

    def test_answer_without_number_rejected(self):
        result = make_completion_result("You need to multiply the values together.", finish_reason="stop")
        assert validate(result, TaskCategory.MATHEMATICAL_REASONING) is False

    def test_just_a_number_accepted(self):
        result = make_completion_result("42", finish_reason="stop")
        assert validate(result, TaskCategory.MATHEMATICAL_REASONING) is True


class TestCodeValidation:
    """Code kategorileri icin kod isareti kontrolu."""

    def test_code_with_def_accepted(self):
        result = make_completion_result("def solution():\n    return 42", finish_reason="stop")
        assert validate(result, TaskCategory.CODE_GENERATION) is True

    def test_code_with_fences_accepted(self):
        result = make_completion_result("```python\ndef f(): pass\n```", finish_reason="stop")
        assert validate(result, TaskCategory.CODE_DEBUGGING) is True

    def test_code_with_return_accepted(self):
        result = make_completion_result("The fix is to use return total instead of total = n", finish_reason="stop")
        assert validate(result, TaskCategory.CODE_DEBUGGING) is True

    def test_prose_without_code_rejected(self):
        result = make_completion_result("You should check the loop variable assignment.", finish_reason="stop")
        assert validate(result, TaskCategory.CODE_GENERATION) is False

"""
test_main.py
=============
main modulu icin birim testleri.

Kontrol edilenler:
  - read_tasks: gecerli/gecersiz input dosyalari
  - write_results: atomik yazma (tmp -> replace)
  - solve_all_tasks: deadline yonetimi, hata izolasyonu
"""

import json
import os
import time

import pytest
from unittest.mock import MagicMock, patch

from main import read_tasks, write_results, solve_all_tasks


class TestReadTasks:
    """read_tasks fonksiyonunun input dosyasini dogru okumasi."""

    def test_valid_tasks_file(self, tmp_path):
        tasks = [
            {"task_id": "t1", "prompt": "What is 2+2?"},
            {"task_id": "t2", "prompt": "Hello."},
        ]
        p = tmp_path / "tasks.json"
        p.write_text(json.dumps(tasks), encoding="utf-8")
        result = read_tasks(str(p))
        assert len(result) == 2
        assert result[0]["task_id"] == "t1"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_tasks("/nonexistent/path/tasks.json")

    def test_not_a_list_raises(self, tmp_path):
        p = tmp_path / "tasks.json"
        p.write_text('{"task_id": "t1"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON listesi"):
            read_tasks(str(p))

    def test_missing_task_id_raises(self, tmp_path):
        p = tmp_path / "tasks.json"
        p.write_text('[{"prompt": "hello"}]', encoding="utf-8")
        with pytest.raises(ValueError, match="task_id"):
            read_tasks(str(p))

    def test_missing_prompt_raises(self, tmp_path):
        p = tmp_path / "tasks.json"
        p.write_text('[{"task_id": "t1"}]', encoding="utf-8")
        with pytest.raises(ValueError, match="prompt"):
            read_tasks(str(p))


class TestWriteResults:
    """write_results fonksiyonunun atomik yazma davranisi."""

    def test_writes_valid_json(self, tmp_path):
        out = tmp_path / "results.json"
        results = [{"task_id": "t1", "answer": "42"}]
        write_results(str(out), results)
        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == results

    def test_no_tmp_file_left(self, tmp_path):
        out = tmp_path / "results.json"
        write_results(str(out), [{"task_id": "t1", "answer": "ok"}])
        assert not os.path.exists(str(out) + ".tmp")

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "results.json"
        write_results(str(out), [{"task_id": "t1", "answer": "ok"}])
        assert out.exists()


class TestSolveAllTasks:
    """solve_all_tasks fonksiyonunun orkestrasyon davranisi."""

    def test_all_tasks_solved(self):
        mock_router = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "answer"
        mock_result.source.value = "default_model"
        mock_result.category.value = "factual_knowledge"
        mock_result.model_used = "minimax-m3"
        mock_result.tokens_spent = 10
        mock_result.was_corrected = False
        mock_router.solve.return_value = mock_result

        tasks = [
            {"task_id": "t1", "prompt": "q1"},
            {"task_id": "t2", "prompt": "q2"},
        ]
        deadline = time.monotonic() + 300  # bol zaman

        results = solve_all_tasks(tasks, mock_router, deadline)
        assert len(results) == 2
        assert results[0]["task_id"] == "t1"
        assert results[0]["answer"] == "answer"

    def test_deadline_fills_empty_answers(self):
        """Deadline gectiyse kalan gorevler bos cevapla doldurulmali."""
        mock_router = MagicMock()
        tasks = [
            {"task_id": "t1", "prompt": "q1"},
            {"task_id": "t2", "prompt": "q2"},
        ]
        deadline = time.monotonic() - 1  # ZATEN gecmis

        results = solve_all_tasks(tasks, mock_router, deadline)
        assert len(results) == 2
        assert results[0]["answer"] == ""
        assert results[1]["answer"] == ""
        mock_router.solve.assert_not_called()

    def test_exception_isolated_per_task(self):
        """Bir gorev patlasa bile digerleri etkilenmemeli."""
        call_count = {"n": 0}

        mock_router = MagicMock()
        def side_effect(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated crash")
            result = MagicMock()
            result.text = "ok"
            result.source.value = "default_model"
            result.category.value = "factual_knowledge"
            result.model_used = "minimax-m3"
            result.tokens_spent = 10
            result.was_corrected = False
            return result

        mock_router.solve.side_effect = side_effect

        tasks = [
            {"task_id": "t1", "prompt": "will crash"},
            {"task_id": "t2", "prompt": "will succeed"},
        ]
        deadline = time.monotonic() + 300

        results = solve_all_tasks(tasks, mock_router, deadline)
        assert len(results) == 2
        assert results[0]["answer"] == ""  # crash olan gorev bos cevap
        assert results[1]["answer"] == "ok"  # digeri basarili

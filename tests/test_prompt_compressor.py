"""
test_prompt_compressor.py
==========================
prompt_compressor modulu icin birim testleri.

Kontrol edilenler:
  - Coklu bosluk/satir sikistirmasi
  - Kod blogu iceriginin BIREBIR korunmasi
  - Bos/None input'larin guvenli islenmesi
"""

from prompt_compressor import compress_prompt


class TestWhitespaceCompression:
    """Bosluk/satir sikistirma davranisi."""

    def test_multiple_spaces_compressed(self):
        assert compress_prompt("hello   world") == "hello world"

    def test_trailing_spaces_before_newline(self):
        assert compress_prompt("hello   \nworld") == "hello\nworld"

    def test_multiple_blank_lines_compressed(self):
        result = compress_prompt("hello\n\n\n\nworld")
        assert result == "hello\n\nworld"

    def test_tabs_compressed(self):
        assert compress_prompt("hello\t\tworld") == "hello world"

    def test_leading_trailing_whitespace_stripped(self):
        assert compress_prompt("  hello world  ") == "hello world"


class TestCodeBlockPreservation:
    """Kod bloklarinin BIREBIR korunmasi - girinti/bosluk semantik tasir."""

    def test_code_block_indentation_preserved(self):
        text = "Here is code:\n```python\ndef f():\n    return 42\n```\nEnd."
        result = compress_prompt(text)
        # Kod blogunun ic kismi DEGISMEMELI
        assert "    return 42" in result

    def test_code_block_multiple_spaces_preserved(self):
        text = "text ```x   =   1``` more text"
        result = compress_prompt(text)
        assert "x   =   1" in result

    def test_surrounding_text_compressed_code_intact(self):
        text = "lots   of    spaces ```  code  ``` more   spaces"
        result = compress_prompt(text)
        assert "lots of spaces" in result
        assert "  code  " in result
        assert "more spaces" in result


class TestEdgeCases:
    """Kenar durumlari."""

    def test_empty_string(self):
        assert compress_prompt("") == ""

    def test_none_passthrough(self):
        # compress_prompt None alirsa None donmeli (not text guard)
        assert compress_prompt(None) is None

    def test_no_code_block_just_text(self):
        result = compress_prompt("simple    text    here")
        assert result == "simple text here"

    def test_only_code_block(self):
        text = "```python\ndef f():\n    pass\n```"
        result = compress_prompt(text)
        assert "    pass" in result

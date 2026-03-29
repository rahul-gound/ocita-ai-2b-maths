"""Tests for the data module."""

from ocita_math.data.dataset import format_math_example, MathDataset


class TestFormatMathExample:
    """Tests for the math example formatting."""

    def test_basic_format(self):
        result = format_math_example("What is 2+2?", "4")
        assert "What is 2+2?" in result
        assert "4" in result
        assert "### Problem:" in result
        assert "### Solution:" in result

    def test_chat_format_with_system_prompt(self):
        result = format_math_example(
            "What is 2+2?", "4", system_prompt="You are a math assistant."
        )
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert "You are a math assistant." in result
        assert "What is 2+2?" in result


class TestMathDataset:
    """Tests for MathDataset."""

    def test_length(self):
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        ds = MathDataset(examples)
        assert len(ds) == 2

    def test_getitem(self):
        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        ds = MathDataset(examples)
        assert ds[0] == {"input_ids": [1, 2, 3]}
        assert ds[1] == {"input_ids": [4, 5, 6]}

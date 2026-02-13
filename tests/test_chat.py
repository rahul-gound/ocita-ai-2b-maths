"""Tests for the chat inference module."""

from ocita_math.inference.chat import format_chat_prompt, SYSTEM_PROMPT


class TestFormatChatPrompt:
    """Tests for chat prompt formatting."""

    def test_default_system_prompt(self):
        prompt = format_chat_prompt("What is 5 + 3?")
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "What is 5 + 3?" in prompt
        assert "Ocita" in prompt
        assert "English" in prompt

    def test_custom_system_prompt(self):
        prompt = format_chat_prompt("Solve x^2 = 4", system_prompt="Custom prompt")
        assert "Custom prompt" in prompt
        assert "Solve x^2 = 4" in prompt

    def test_system_prompt_mentions_english(self):
        assert "English" in SYSTEM_PROMPT

    def test_system_prompt_mentions_math(self):
        assert "math" in SYSTEM_PROMPT.lower()

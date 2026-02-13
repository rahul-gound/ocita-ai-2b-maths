"""Tests for the Ocita 2B Maths model configuration."""

import os
import tempfile

import yaml

from ocita_math.model.config import ModelConfig, TrainingConfig, ChatConfig


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        config = ModelConfig()
        assert config.name == "ocita-2b-maths"
        assert config.vocab_size == 50304
        assert config.hidden_size == 2048
        assert config.intermediate_size == 8192
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 16
        assert config.max_position_embeddings == 2048
        assert config.tie_word_embeddings is True

    def test_head_dim(self):
        config = ModelConfig()
        assert config.head_dim == 128  # 2048 / 16

    def test_param_count_approximately_2b(self):
        config = ModelConfig()
        param_count = config.param_count_estimate()
        assert 1_500_000_000 < param_count < 2_500_000_000, (
            f"Expected ~2B parameters, got {param_count:,}"
        )

    def test_from_yaml(self):
        yaml_content = {
            "model": {
                "name": "test-model",
                "vocab_size": 1000,
                "hidden_size": 128,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "max_position_embeddings": 256,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()
            config = ModelConfig.from_yaml(f.name)

        os.unlink(f.name)
        assert config.name == "test-model"
        assert config.vocab_size == 1000
        assert config.hidden_size == 128
        assert config.num_hidden_layers == 2


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        config = TrainingConfig()
        assert config.learning_rate == 3e-4
        assert config.max_steps == 100000
        assert config.batch_size == 32
        assert config.bf16 is True
        assert config.seed == 42


class TestChatConfig:
    """Tests for ChatConfig."""

    def test_default_config(self):
        config = ChatConfig()
        assert "Ocita" in config.system_prompt
        assert "English" in config.system_prompt
        assert "math" in config.system_prompt.lower()
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.7

"""Tests for the Ocita 2B Maths model architecture."""

import torch

from ocita_math.model.architecture import (
    OcitaMathModel,
    RMSNorm,
    SwiGLUFFN,
    MathAttention,
    TransformerBlock,
)
from ocita_math.model.config import ModelConfig


def _small_config() -> ModelConfig:
    """Create a small model config for fast testing."""
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_output(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestMathAttention:
    """Tests for MathAttention with rotary embeddings."""

    def test_output_shape(self):
        config = _small_config()
        attn = MathAttention(config)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_causal_attention(self):
        config = _small_config()
        attn = MathAttention(config)
        x = torch.randn(1, 4, 64)
        mask = torch.triu(torch.ones(4, 4), diagonal=1).masked_fill(
            torch.triu(torch.ones(4, 4), diagonal=1) == 1, float("-inf")
        ).unsqueeze(0).unsqueeze(0)
        out = attn(x, attention_mask=mask)
        assert out.shape == (1, 4, 64)


class TestSwiGLUFFN:
    """Tests for SwiGLU feed-forward network."""

    def test_output_shape(self):
        config = _small_config()
        ffn = SwiGLUFFN(config)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)


class TestTransformerBlock:
    """Tests for a single transformer block."""

    def test_output_shape(self):
        config = _small_config()
        block = TransformerBlock(config)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)


class TestOcitaMathModel:
    """Tests for the full Ocita Math model."""

    def test_forward_pass(self):
        config = _small_config()
        model = OcitaMathModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 16, config.vocab_size)

    def test_forward_with_labels(self):
        config = _small_config()
        model = OcitaMathModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = input_ids.clone()
        outputs = model(input_ids, labels=labels)
        assert "loss" in outputs
        assert outputs["loss"].item() > 0

    def test_parameter_count(self):
        config = _small_config()
        model = OcitaMathModel(config)
        param_count = model.count_parameters()
        assert param_count > 0

    def test_full_model_approximately_2b(self):
        """Verify the default config produces a ~2B parameter model."""
        config = ModelConfig()
        estimated = config.param_count_estimate()
        assert 1_500_000_000 < estimated < 2_500_000_000, (
            f"Expected ~2B parameters, got {estimated:,}"
        )

    def test_generate(self):
        config = _small_config()
        model = OcitaMathModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        generated = model.generate(input_ids, max_new_tokens=8, temperature=0.7)
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 4  # At least the input length
        assert generated.shape[1] <= 12  # At most input + max_new_tokens

    def test_generate_deterministic(self):
        config = _small_config()
        model = OcitaMathModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        generated = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
        assert generated.shape[1] == 9  # 4 input + 5 new tokens

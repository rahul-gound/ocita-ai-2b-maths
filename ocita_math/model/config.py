"""Model configuration for Ocita AI 2B Maths."""

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for the Ocita 2B math transformer model.

    Default parameters produce a model with approximately 2 billion parameters:
    - Embedding: vocab_size * hidden_size = 50304 * 2048 ≈ 103M
    - Per layer: ~52M (attention + FFN + norms)
    - 32 layers: 32 * 52M ≈ 1,664M
    - LM head (tied): 0 extra
    - Total: ~1,767M + embeddings ≈ ~2B parameters
    """

    name: str = "ocita-2b-maths"
    vocab_size: int = 50304
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    max_position_embeddings: int = 2048
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_rotary_embeddings: bool = True
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def param_count_estimate(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.hidden_size
        attn_per_layer = 4 * self.hidden_size * self.hidden_size
        ffn_per_layer = 2 * self.hidden_size * self.intermediate_size
        norm_per_layer = 2 * self.hidden_size
        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        total_layers = self.num_hidden_layers * per_layer
        final_norm = self.hidden_size
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        return embed + total_layers + final_norm + lm_head

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load model configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        model_cfg = raw.get("model", {})
        return cls(**{k: v for k, v in model_cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    lr_scheduler: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load training configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        training_cfg = raw.get("training", {})
        return cls(**{k: v for k, v in training_cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class ChatConfig:
    """Chat/inference configuration."""

    system_prompt: str = (
        "You are Ocita, a helpful AI math assistant. You solve math problems "
        "step by step, showing clear reasoning. You communicate in English only. "
        "When presented with a math problem, break it down into steps and provide "
        "the final answer clearly."
    )
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    @classmethod
    def from_yaml(cls, path: str) -> "ChatConfig":
        """Load chat configuration from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        chat_cfg = raw.get("chat", {})
        return cls(**{k: v for k, v in chat_cfg.items() if k in cls.__dataclass_fields__})

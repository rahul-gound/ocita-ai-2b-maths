# Ocita AI 2B Maths

A **2 billion parameter** AI model built for solving math problems, with an **English-only** chat interface.

## Overview

Ocita AI 2B Maths is a decoder-only transformer language model (~2B parameters) designed and trained specifically for mathematical reasoning. It communicates in English and provides step-by-step solutions to math problems.

### Key Features

- **~2 billion parameters** – GPT-style transformer with 32 layers, 16 attention heads, and hidden size 2048
- **Math-focused** – Trained on curated math datasets (GSM8K, MATH, competition problems)
- **English chat interface** – Interactive CLI for asking math questions in English
- **Modern architecture** – RoPE positional embeddings, SwiGLU activations, RMSNorm, pre-norm design
- **Mixed precision training** – BF16/FP16 support for efficient training

### Model Architecture

| Parameter | Value |
|---|---|
| Parameters | ~2B |
| Layers | 32 |
| Hidden Size | 2048 |
| FFN Size | 8192 |
| Attention Heads | 16 |
| Head Dim | 128 |
| Vocabulary | 50,304 |
| Max Sequence Length | 2,048 |
| Positional Encoding | RoPE |
| Activation | SwiGLU |
| Normalization | RMSNorm |

## Project Structure

```
ocita-ai-2b-maths/
├── config/
│   └── model_config.yaml        # Model, training, and chat configuration
├── ocita_math/
│   ├── model/
│   │   ├── config.py            # Model/training/chat config dataclasses
│   │   └── architecture.py      # 2B transformer model definition
│   ├── data/
│   │   └── dataset.py           # Math dataset loading and preprocessing
│   ├── training/
│   │   └── train.py             # Training loop with mixed precision
│   └── inference/
│       └── chat.py              # English chat interface
├── tests/                       # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on math datasets:

```bash
# Train with default configuration
python -m ocita_math.training.train --dataset gsm8k --output-dir checkpoints

# Train with custom YAML config
python -m ocita_math.training.train --config config/model_config.yaml --output-dir checkpoints

# Resume training from checkpoint
python -m ocita_math.training.train --resume checkpoints/checkpoint-5000.pt
```

### Chat Interface

Start an interactive English math chat:

```bash
python -m ocita_math.inference.chat --checkpoint checkpoints/model_final.pt
```

Example session:

```
============================================================
  Ocita AI 2B Maths - Interactive Chat
  Model parameters: 1,965,490,176 (~1.97B)
  Language: English only
  Type 'quit' or 'exit' to end the session
============================================================

You: What is the derivative of x^3 + 2x^2 - 5x + 3?
Ocita: Let me solve this step by step.

Step 1: Apply the power rule to each term.
  - d/dx(x^3) = 3x^2
  - d/dx(2x^2) = 4x
  - d/dx(-5x) = -5
  - d/dx(3) = 0

Step 2: Combine the results.
  3x^2 + 4x - 5

The derivative is **3x² + 4x - 5**.
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Configuration

All model, training, and inference settings are in `config/model_config.yaml`. Key sections:

- **model**: Architecture hyperparameters (layers, hidden size, heads, etc.)
- **training**: Learning rate, batch size, optimizer settings, mixed precision
- **chat**: System prompt, generation parameters (temperature, top-p, top-k)

## License

MIT
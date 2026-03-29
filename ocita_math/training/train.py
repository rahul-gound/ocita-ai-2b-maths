"""Training script for the Ocita AI 2B Maths model.

Handles the full training pipeline including:
- Model initialization with ~2B parameters
- Data loading and preprocessing
- Distributed training with mixed precision
- Checkpointing and logging
"""

import argparse
import logging
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ocita_math.model.architecture import OcitaMathModel
from ocita_math.model.config import ModelConfig, TrainingConfig, ChatConfig
from ocita_math.data.dataset import load_and_tokenize_dataset

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay exclusion for norms and biases."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.max_steps - config.warmup_steps
        )
        min_ratio = config.min_learning_rate / config.learning_rate
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model_config_path: Optional[str] = None,
    dataset_name: str = "gsm8k",
    output_dir: str = "checkpoints",
    resume_from: Optional[str] = None,
) -> None:
    """Run the training loop.

    Args:
        model_config_path: Path to YAML config file.
        dataset_name: HuggingFace dataset name for math data.
        output_dir: Directory for saving checkpoints.
        resume_from: Path to checkpoint to resume training from.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if model_config_path:
        model_config = ModelConfig.from_yaml(model_config_path)
        training_config = TrainingConfig.from_yaml(model_config_path)
        chat_config = ChatConfig.from_yaml(model_config_path)
    else:
        model_config = ModelConfig()
        training_config = TrainingConfig()
        chat_config = ChatConfig()

    torch.manual_seed(training_config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Initializing Ocita 2B Math model...")
    model = OcitaMathModel(model_config)
    param_count = model.count_parameters()
    logger.info("Model parameters: %s (~%.2fB)", f"{param_count:,}", param_count / 1e9)

    if resume_from and os.path.exists(resume_from):
        logger.info("Resuming from checkpoint: %s", resume_from)
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    dtype = torch.bfloat16 if training_config.bf16 else (torch.float16 if training_config.fp16 else torch.float32)

    optimizer = create_optimizer(model, training_config)
    scheduler = get_lr_scheduler(optimizer, training_config)

    if resume_from and os.path.exists(resume_from):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0)
    else:
        start_step = 0

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset: %s", dataset_name)
    train_dataset = load_and_tokenize_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_length=model_config.max_position_embeddings,
        split="train",
        system_prompt=chat_config.system_prompt,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Starting training for %d steps...", training_config.max_steps)
    model.train()
    global_step = start_step
    running_loss = 0.0

    while global_step < training_config.max_steps:
        for batch in dataloader:
            if global_step >= training_config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / training_config.gradient_accumulation_steps

            loss.backward()

            if (global_step + 1) % training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = running_loss / 100
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Step %d/%d | Loss: %.4f | LR: %.2e",
                    global_step, training_config.max_steps, avg_loss, lr,
                )
                running_loss = 0.0

            if global_step % 5000 == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": model_config,
                    },
                    checkpoint_path,
                )
                logger.info("Saved checkpoint: %s", checkpoint_path)

    final_path = os.path.join(output_dir, "model_final.pt")
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "config": model_config,
        },
        final_path,
    )
    logger.info("Training complete. Final model saved to: %s", final_path)


def main() -> None:
    """Entry point for training CLI."""
    parser = argparse.ArgumentParser(description="Train Ocita AI 2B Maths Model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="HuggingFace dataset name")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    train(
        model_config_path=args.config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()

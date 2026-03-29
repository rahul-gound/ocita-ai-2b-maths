"""English-language chat interface for the Ocita AI 2B Maths model.

Provides an interactive command-line chat where users can ask math problems
in English and receive step-by-step solutions.
"""

import argparse
import logging
import sys
from typing import Optional

import torch

from ocita_math.model.architecture import OcitaMathModel
from ocita_math.model.config import ModelConfig, ChatConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Ocita, a helpful AI math assistant. You solve math problems "
    "step by step, showing clear reasoning. You communicate in English only. "
    "When presented with a math problem, break it down into steps and provide "
    "the final answer clearly."
)


def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a trained Ocita model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint.
        device: Target device for the model.

    Returns:
        Tuple of (model, config).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", ModelConfig())
    model = OcitaMathModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def format_chat_prompt(
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Format user input into a chat prompt for the model.

    Args:
        user_message: The user's math question in English.
        system_prompt: System instruction for the model.

    Returns:
        Formatted prompt string.
    """
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n"
    )


def generate_response(
    model: OcitaMathModel,
    tokenizer,
    user_message: str,
    chat_config: Optional[ChatConfig] = None,
    device: Optional[torch.device] = None,
) -> str:
    """Generate a math solution response for the given user message.

    Args:
        model: The loaded Ocita model.
        tokenizer: The tokenizer.
        user_message: User's math question in English.
        chat_config: Chat generation parameters.
        device: Target device.

    Returns:
        Model's response string.
    """
    if chat_config is None:
        chat_config = ChatConfig()
    if device is None:
        device = next(model.parameters()).device

    prompt = format_chat_prompt(user_message, chat_config.system_prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=chat_config.max_new_tokens,
        temperature=chat_config.temperature,
        top_p=chat_config.top_p,
        top_k=chat_config.top_k,
        repetition_penalty=chat_config.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def interactive_chat(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> None:
    """Run interactive English math chat in the terminal.

    Args:
        checkpoint_path: Path to the trained model checkpoint.
        config_path: Optional path to YAML config file.
    """
    logging.basicConfig(level=logging.INFO)

    if config_path:
        chat_config = ChatConfig.from_yaml(config_path)
    else:
        chat_config = ChatConfig()

    from transformers import AutoTokenizer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from: %s", checkpoint_path)
    model, model_config = load_model(checkpoint_path)

    param_count = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"  Ocita AI 2B Maths - Interactive Chat")
    print(f"  Model parameters: {param_count:,} (~{param_count/1e9:.2f}B)")
    print(f"  Language: English only")
    print(f"  Type 'quit' or 'exit' to end the session")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("Ocita: Thinking...", end="\r")
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_message=user_input,
            chat_config=chat_config,
        )
        print(f"Ocita: {response}\n")


def main() -> None:
    """Entry point for chat CLI."""
    parser = argparse.ArgumentParser(
        description="Ocita AI 2B Maths - English Math Chat Interface"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    interactive_chat(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

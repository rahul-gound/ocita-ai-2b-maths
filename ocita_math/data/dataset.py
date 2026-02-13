"""Math dataset loading and preprocessing for Ocita AI training.

Supports loading math-specific datasets (GSM8K, MATH, etc.) and formatting
them into a consistent prompt-completion format for training.
"""

from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


MATH_PROMPT_TEMPLATE = (
    "### Problem:\n{question}\n\n### Solution:\n{answer}"
)

CHAT_PROMPT_TEMPLATE = (
    "<|system|>\n{system}\n<|user|>\n{question}\n<|assistant|>\n{answer}"
)


def format_math_example(
    question: str,
    answer: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Format a single math example into the training prompt format.

    Args:
        question: The math problem text.
        answer: The step-by-step solution.
        system_prompt: Optional system prompt for chat format.

    Returns:
        Formatted prompt string.
    """
    if system_prompt:
        return CHAT_PROMPT_TEMPLATE.format(
            system=system_prompt, question=question, answer=answer
        )
    return MATH_PROMPT_TEMPLATE.format(question=question, answer=answer)


class MathDataset(Dataset):
    """Dataset for math problem-solution pairs.

    Wraps tokenized math examples for training. Each example is a dictionary
    with 'input_ids', 'attention_mask', and 'labels' tensors.
    """

    def __init__(self, examples: List[Dict[str, Any]]):
        """Initialize with a list of tokenized examples.

        Args:
            examples: List of dicts with 'input_ids', 'attention_mask', 'labels'.
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def load_and_tokenize_dataset(
    dataset_name: str,
    tokenizer: Any,
    max_length: int = 2048,
    split: str = "train",
    system_prompt: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> MathDataset:
    """Load a math dataset from HuggingFace and tokenize it.

    Args:
        dataset_name: Name of the HuggingFace dataset (e.g., 'gsm8k').
        tokenizer: The tokenizer to use for encoding.
        max_length: Maximum sequence length.
        split: Dataset split to load.
        system_prompt: Optional system prompt to prepend.
        max_samples: Optional limit on number of samples.

    Returns:
        MathDataset containing tokenized examples.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    question_key = _detect_question_key(dataset)
    answer_key = _detect_answer_key(dataset)

    examples = []
    for item in dataset:
        text = format_math_example(
            question=item[question_key],
            answer=item[answer_key],
            system_prompt=system_prompt,
        )
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        example = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone(),
        }
        examples.append(example)

    return MathDataset(examples)


def _detect_question_key(dataset: Any) -> str:
    """Detect the question field name in a dataset."""
    for key in ["question", "problem", "input", "prompt", "query"]:
        if key in dataset.column_names:
            return key
    return dataset.column_names[0]


def _detect_answer_key(dataset: Any) -> str:
    """Detect the answer field name in a dataset."""
    for key in ["answer", "solution", "output", "response", "target"]:
        if key in dataset.column_names:
            return key
    return dataset.column_names[-1]

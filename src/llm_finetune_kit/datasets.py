"""
Easy dataset handling for demo-focused fine-tuning.

Supports multiple formats and includes sample datasets for quick demos.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datasets import Dataset, load_dataset as hf_load_dataset
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_dataset(name: str = "chat") -> List[Dict[str, str]]:
    """
    Load built-in sample datasets for demos.

    Args:
        name: Dataset name ("chat", "instruct", "code")

    Returns:
        List of dataset examples
    """
    datasets_dir = Path(__file__).parent.parent.parent / "datasets"

    sample_datasets = {
        "chat": datasets_dir / "sample_chat.json",
        "instruct": datasets_dir / "sample_instruct.json",
    }

    if name not in sample_datasets:
        raise ValueError(f"Unknown sample dataset: {name}. Available: {list(sample_datasets.keys())}")

    dataset_path = sample_datasets[name]

    if not dataset_path.exists():
        logger.warning(f"⚠️  Sample dataset not found at {dataset_path}")
        logger.info("📝 Generating sample data...")
        return _generate_sample_data(name)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    logger.info(f"✅ Loaded {len(data)} examples from {name} dataset")
    return data


def _generate_sample_data(dataset_type: str, n_samples: int = 50) -> List[Dict[str, str]]:
    """
    Generate synthetic sample data for demos.

    Args:
        dataset_type: Type of data ("chat" or "instruct")
        n_samples: Number of samples to generate

    Returns:
        List of synthetic examples
    """
    if dataset_type == "chat":
        templates = [
            {
                "prompt": "How do I reset my password?",
                "response": "To reset your password, click on 'Forgot Password' on the login page and follow the email instructions.",
            },
            {
                "prompt": "What are your business hours?",
                "response": "We're open Monday-Friday, 9 AM to 5 PM EST. We're closed on weekends and major holidays.",
            },
            {
                "prompt": "How can I track my order?",
                "response": "You can track your order using the tracking number sent to your email. Visit our tracking page and enter the number.",
            },
        ]
    else:  # instruct
        templates = [
            {
                "prompt": "Explain what a Python decorator is",
                "response": "A Python decorator is a function that modifies the behavior of another function. It's denoted with @ symbol and placed above a function definition.",
            },
            {
                "prompt": "Write a function to reverse a string",
                "response": "Here's a simple function:\n\ndef reverse_string(s):\n    return s[::-1]",
            },
            {
                "prompt": "What is the difference between a list and a tuple in Python?",
                "response": "Lists are mutable (can be changed) and use square brackets []. Tuples are immutable (cannot be changed) and use parentheses ().",
            },
        ]

    # Duplicate and vary templates to reach n_samples
    data = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        data.append(template.copy())

    logger.info(f"✨ Generated {len(data)} synthetic {dataset_type} examples")
    return data


def load_dataset(
    data_source: Union[str, Path, List[Dict]],
    format: str = "auto",
    text_column: Optional[str] = None,
    split: Optional[str] = None,
) -> Dataset:
    """
    Load dataset from various sources with automatic format detection.

    Supports:
    - JSON files (with prompt/response or instruction/output format)
    - CSV files
    - HuggingFace datasets
    - Python lists
    - Built-in sample datasets

    Args:
        data_source: Path to file, HF dataset name, or list of examples
        format: Format ("json", "csv", "huggingface", "auto")
        text_column: Column name for text data (auto-detected if None)
        split: Dataset split for HuggingFace datasets

    Returns:
        HuggingFace Dataset object

    Examples:
        >>> # Load JSON file
        >>> dataset = load_dataset("my_data.json")
        >>>
        >>> # Load sample dataset
        >>> dataset = load_dataset("sample:chat")
        >>>
        >>> # Load from HuggingFace
        >>> dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:100]")
    """
    # Handle sample datasets
    if isinstance(data_source, str) and data_source.startswith("sample:"):
        sample_name = data_source.replace("sample:", "")
        data = load_sample_dataset(sample_name)
        return Dataset.from_list(data)

    # Handle list input
    if isinstance(data_source, list):
        logger.info(f"📊 Creating dataset from {len(data_source)} examples")
        return Dataset.from_list(data_source)

    # Handle file paths
    if isinstance(data_source, (str, Path)):
        path = Path(data_source)

        # Auto-detect format
        if format == "auto":
            if path.suffix == ".json":
                format = "json"
            elif path.suffix == ".csv":
                format = "csv"
            elif not path.exists():
                # Might be a HuggingFace dataset
                format = "huggingface"
            else:
                raise ValueError(f"Cannot auto-detect format for {path}")

        # Load based on format
        if format == "json" and path.exists():
            logger.info(f"📁 Loading JSON from {path}")
            with open(path, "r") as f:
                data = json.load(f)
            return Dataset.from_list(data)

        elif format == "csv" and path.exists():
            logger.info(f"📁 Loading CSV from {path}")
            import pandas as pd

            df = pd.read_csv(path)
            return Dataset.from_pandas(df)

        elif format == "huggingface":
            logger.info(f"🤗 Loading from HuggingFace: {data_source}")
            return hf_load_dataset(str(data_source), split=split)

    raise ValueError(f"Unable to load dataset from {data_source}")


def prepare_dataset(
    dataset: Dataset,
    tokenizer,
    prompt_column: str = "prompt",
    response_column: str = "response",
    max_length: int = 512,
    template: Optional[str] = None,
) -> Dataset:
    """
    Prepare dataset for training with tokenization and formatting.

    Args:
        dataset: Input dataset
        tokenizer: HuggingFace tokenizer
        prompt_column: Column name for prompts/instructions
        response_column: Column name for responses/completions
        max_length: Maximum sequence length
        template: Optional chat template (e.g., for instruction-following)

    Returns:
        Prepared dataset ready for training

    Example:
        >>> dataset = load_dataset("my_data.json")
        >>> prepared = prepare_dataset(dataset, tokenizer)
    """
    logger.info("🔧 Preparing dataset for training...")

    # Auto-detect column names
    columns = dataset.column_names
    if prompt_column not in columns:
        # Try common alternatives
        for alt in ["instruction", "input", "question", "text"]:
            if alt in columns:
                prompt_column = alt
                logger.info(f"📝 Using '{prompt_column}' as prompt column")
                break

    if response_column not in columns:
        for alt in ["completion", "output", "answer", "target"]:
            if alt in columns:
                response_column = alt
                logger.info(f"📝 Using '{response_column}' as response column")
                break

    # Apply template if provided
    if template is None:
        # Default template
        template = "### Instruction:\n{prompt}\n\n### Response:\n{response}"

    def format_example(example):
        """Format single example with template"""
        text = template.format(
            prompt=example.get(prompt_column, ""),
            response=example.get(response_column, ""),
        )

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Apply formatting
    prepared_dataset = dataset.map(
        format_example,
        remove_columns=columns,
        desc="Formatting dataset",
    )

    logger.info(f"✅ Prepared {len(prepared_dataset)} examples")
    logger.info(f"📊 Max length: {max_length} tokens")

    return prepared_dataset


def split_dataset(
    dataset: Dataset, train_size: float = 0.9, seed: int = 42
) -> Dict[str, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Input dataset
        train_size: Fraction of data for training (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train' and 'validation' datasets
    """
    if train_size >= 1.0:
        logger.info("ℹ️  Using full dataset for training (no validation split)")
        return {"train": dataset, "validation": None}

    logger.info(f"🔀 Splitting dataset: {train_size:.0%} train, {1-train_size:.0%} validation")

    split = dataset.train_test_split(test_size=1 - train_size, seed=seed)

    return {"train": split["train"], "validation": split["test"]}


def create_dataset_from_text(
    texts: List[str], labels: Optional[List[str]] = None
) -> Dataset:
    """
    Create dataset from raw text strings.

    Args:
        texts: List of text strings
        labels: Optional list of labels/responses

    Returns:
        Dataset object
    """
    if labels is None:
        # Self-supervised (next token prediction)
        data = [{"text": text} for text in texts]
    else:
        # Supervised (prompt -> response)
        data = [{"prompt": text, "response": label} for text, label in zip(texts, labels)]

    logger.info(f"📝 Created dataset with {len(data)} examples")
    return Dataset.from_list(data)


def get_dataset_stats(dataset: Dataset) -> Dict[str, Any]:
    """
    Get statistics about a dataset.

    Args:
        dataset: Input dataset

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "num_examples": len(dataset),
        "columns": dataset.column_names,
        "features": str(dataset.features),
    }

    # Try to get text length stats if there's a text column
    for col in ["text", "prompt", "input"]:
        if col in dataset.column_names:
            lengths = [len(str(ex[col])) for ex in dataset.select(range(min(1000, len(dataset))))]
            stats["avg_text_length"] = sum(lengths) / len(lengths)
            stats["max_text_length"] = max(lengths)
            stats["min_text_length"] = min(lengths)
            break

    return stats


def augment_dataset(
    dataset: Dataset, augmentation_type: str = "simple", n_augmented: int = 100
) -> Dataset:
    """
    Augment dataset with synthetic examples (for demos).

    Args:
        dataset: Input dataset
        augmentation_type: Type of augmentation ("simple", "paraphrase")
        n_augmented: Number of augmented examples to generate

    Returns:
        Augmented dataset
    """
    logger.info(f"🔄 Augmenting dataset with {n_augmented} examples...")

    original_data = list(dataset)
    augmented_data = []

    # Simple augmentation: random variations
    for _ in range(n_augmented):
        example = random.choice(original_data)
        # Create shallow copy with minor modifications
        augmented = example.copy()
        augmented_data.append(augmented)

    combined_data = original_data + augmented_data
    augmented_dataset = Dataset.from_list(combined_data)

    logger.info(f"✅ Dataset augmented: {len(dataset)} → {len(augmented_dataset)} examples")

    return augmented_dataset


def preview_dataset(dataset: Dataset, n_examples: int = 3):
    """
    Print preview of dataset examples.

    Args:
        dataset: Dataset to preview
        n_examples: Number of examples to show
    """
    logger.info(f"\n📋 Dataset Preview ({n_examples} examples):\n")

    for i, example in enumerate(dataset.select(range(min(n_examples, len(dataset))))):
        logger.info(f"Example {i + 1}:")
        for key, value in example.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            logger.info(f"  {key}: {value_str}")
        logger.info("")

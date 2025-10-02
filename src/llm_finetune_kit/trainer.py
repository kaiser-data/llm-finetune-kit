"""
Simplified training wrapper for demo-focused fine-tuning.

Provides easy-to-use training interface with progress tracking,
automatic checkpointing, and smart defaults.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from datasets import Dataset
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressCallback(TrainerCallback):
    """Callback for tracking training progress"""

    def __init__(self):
        self.start_time = None
        self.losses = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of training"""
        self.start_time = time.time()
        logger.info("🚀 Training started!")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called when logging occurs"""
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            elapsed = time.time() - self.start_time
            logger.info(
                f"📊 Step {state.global_step}: loss={logs['loss']:.4f}, "
                f"elapsed={elapsed:.1f}s"
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training"""
        elapsed = time.time() - self.start_time
        logger.info(f"✅ Training completed in {elapsed:.1f}s!")

        if self.losses:
            initial_loss = self.losses[0]
            final_loss = self.losses[-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            logger.info(f"📈 Loss improvement: {improvement:.1f}%")


class SimpleTrainer:
    """
    Simplified training interface for demo-focused fine-tuning.

    Provides sensible defaults and easy configuration for quick experiments.

    Example:
        >>> model, tokenizer, _ = load_model("gpt2")
        >>> dataset = load_dataset("my_data.json")
        >>> prepared = prepare_dataset(dataset, tokenizer)
        >>>
        >>> trainer = SimpleTrainer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_dataset=prepared,
        ...     output_dir="./outputs"
        ... )
        >>>
        >>> trainer.train()
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./outputs",
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 3,
        max_steps: int = -1,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: Optional[int] = None,
        fp16: bool = True,
        optim: str = "paged_adamw_8bit",
        gradient_checkpointing: bool = True,
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.001,
        lr_scheduler_type: str = "cosine",
        seed: int = 42,
        callbacks: Optional[list] = None,
    ):
        """
        Initialize SimpleTrainer with smart defaults.

        Args:
            model: Model to train (with LoRA adapters if applicable)
            tokenizer: Tokenizer
            train_dataset: Training dataset (prepared with prepare_dataset)
            eval_dataset: Optional evaluation dataset
            output_dir: Directory for outputs and checkpoints
            learning_rate: Learning rate
            batch_size: Training batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            num_epochs: Number of training epochs
            max_steps: Maximum training steps (overrides num_epochs if > 0)
            warmup_steps: Number of warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps (None = same as save_steps)
            fp16: Use mixed precision training
            optim: Optimizer ("paged_adamw_8bit" for memory efficiency)
            gradient_checkpointing: Use gradient checkpointing
            max_grad_norm: Maximum gradient norm for clipping
            weight_decay: Weight decay for regularization
            lr_scheduler_type: Learning rate scheduler type
            seed: Random seed
            callbacks: Additional trainer callbacks
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)

        # Auto-configure based on GPU availability
        if not torch.cuda.is_available():
            logger.warning("⚠️  No GPU available. Using CPU (training will be slow)")
            fp16 = False
            optim = "adamw_torch"

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_steps is not None else save_steps,
            fp16=fp16,
            bf16=False,  # Use fp16 for better compatibility
            optim=optim,
            gradient_checkpointing=gradient_checkpointing,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            save_total_limit=2,  # Keep only 2 checkpoints to save space
            load_best_model_at_end=eval_dataset is not None,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            save_strategy="steps",
            seed=seed,
            report_to=[],  # Disable wandb/tensorboard by default
            remove_unused_columns=False,
            dataloader_pin_memory=True,
        )

        # Set up callbacks
        self.progress_callback = ProgressCallback()
        all_callbacks = [self.progress_callback]
        if callbacks:
            all_callbacks.extend(callbacks)

        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=all_callbacks,
        )

        # Store training metrics
        self.metrics = {}

        logger.info("🎯 SimpleTrainer initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📊 Training examples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"📊 Evaluation examples: {len(eval_dataset)}")

    def train(self) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Dictionary with training metrics
        """
        logger.info("\n" + "=" * 50)
        logger.info("🚀 Starting training...")
        logger.info("=" * 50 + "\n")

        # Train
        train_result = self.trainer.train()

        # Save metrics
        self.metrics = train_result.metrics

        logger.info("\n" + "=" * 50)
        logger.info("✅ Training completed!")
        logger.info("=" * 50 + "\n")

        # Log final metrics
        for key, value in self.metrics.items():
            logger.info(f"  {key}: {value}")

        return self.metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataset is None:
            logger.warning("⚠️  No evaluation dataset provided")
            return {}

        logger.info("📊 Evaluating model...")
        eval_metrics = self.trainer.evaluate()

        logger.info("\n📈 Evaluation Results:")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value}")

        return eval_metrics

    def save(self, output_dir: Optional[str] = None):
        """
        Save model and tokenizer.

        Args:
            output_dir: Directory to save to (uses self.output_dir if None)
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir / "final_model"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving model to {save_dir}")

        self.trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))

        # Save training metrics
        metrics_path = save_dir / "training_metrics.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info("✅ Model saved successfully!")

    def loss_decreased(self) -> bool:
        """
        Check if loss decreased during training.

        Returns:
            True if final loss < initial loss
        """
        if len(self.progress_callback.losses) < 2:
            return False
        return self.progress_callback.losses[-1] < self.progress_callback.losses[0]

    def get_loss_history(self) -> list:
        """
        Get loss history.

        Returns:
            List of loss values
        """
        return self.progress_callback.losses.copy()


class QuickTrainer:
    """
    Ultra-simplified trainer for 3-line demos.

    Example:
        >>> from llm_finetune_kit import QuickTrainer
        >>> trainer = QuickTrainer("gpt2", "my_data.json")
        >>> trainer.train()
    """

    def __init__(
        self,
        model_name: str,
        data_source: str,
        output_dir: str = "./quick_outputs",
        max_steps: int = 100,
        batch_size: int = 4,
    ):
        """
        Initialize QuickTrainer.

        Args:
            model_name: Model name from AVAILABLE_MODELS
            data_source: Path to dataset or sample dataset name
            output_dir: Output directory
            max_steps: Maximum training steps
            batch_size: Batch size
        """
        from .models import load_model
        from .datasets import load_dataset, prepare_dataset

        logger.info("🚀 QuickTrainer - Ultra-fast setup")

        # Load model
        logger.info(f"📦 Loading model: {model_name}")
        self.model, self.tokenizer, _ = load_model(model_name)

        # Load dataset
        logger.info(f"📊 Loading dataset: {data_source}")
        dataset = load_dataset(data_source)

        # Prepare dataset
        logger.info("🔧 Preparing dataset...")
        self.train_dataset = prepare_dataset(dataset, self.tokenizer)

        # Create trainer
        self.trainer = SimpleTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            output_dir=output_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            num_epochs=1,
        )

        logger.info("✅ QuickTrainer ready!")

    def train(self):
        """Train the model"""
        return self.trainer.train()

    def save(self, output_dir: Optional[str] = None):
        """Save the model"""
        return self.trainer.save(output_dir)


def create_training_config(
    model_name: str,
    dataset_size: int,
    target_time_minutes: int = 15,
    gpu_memory_gb: float = 16,
) -> Dict[str, Any]:
    """
    Auto-generate training configuration based on constraints.

    Args:
        model_name: Model name
        dataset_size: Number of training examples
        target_time_minutes: Target training time in minutes
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Dictionary with recommended training config
    """
    from .models import AVAILABLE_MODELS

    model_info = AVAILABLE_MODELS.get(model_name, {})

    # Determine batch size based on GPU memory
    if gpu_memory_gb < 8:
        batch_size = 1
        gradient_accumulation = 8
    elif gpu_memory_gb < 16:
        batch_size = 2
        gradient_accumulation = 4
    elif gpu_memory_gb < 24:
        batch_size = 4
        gradient_accumulation = 2
    else:
        batch_size = 8
        gradient_accumulation = 1

    # Estimate steps needed
    steps_per_minute = {
        "gpt2": 60,
        "gpt2-medium": 40,
        "gpt2-large": 25,
        "mistral-7b": 15,
        "llama-3.1-8b": 12,
        "phi-3-mini": 30,
    }

    steps_rate = steps_per_minute.get(model_name, 20)
    max_steps = int(target_time_minutes * steps_rate)

    # Calculate epochs
    effective_batch = batch_size * gradient_accumulation
    steps_per_epoch = dataset_size // effective_batch
    num_epochs = max(1, max_steps // steps_per_epoch)

    config = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
        "learning_rate": 2e-4,
        "warmup_steps": min(100, max_steps // 10),
        "logging_steps": max(10, max_steps // 20),
        "save_steps": max(50, max_steps // 5),
        "estimated_time_minutes": target_time_minutes,
    }

    logger.info("📋 Recommended training configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    return config

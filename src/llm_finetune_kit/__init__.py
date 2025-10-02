"""
LLM Finetune Kit - Beginner-friendly library for fine-tuning small-to-medium LLMs

A demo-focused, production-quality Python library for quick LLM fine-tuning experiments.
Perfect for learning, prototyping, and portfolio demonstrations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models import load_model, get_available_models
from .datasets import load_dataset, prepare_dataset
from .trainer import SimpleTrainer
from .evaluate import evaluate_model, compare_models

__all__ = [
    "load_model",
    "get_available_models",
    "load_dataset",
    "prepare_dataset",
    "SimpleTrainer",
    "evaluate_model",
    "compare_models",
]

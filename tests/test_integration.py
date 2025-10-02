"""
Integration tests for LLM Finetune Kit

Tests the complete training pipeline on a tiny dataset.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import json


def test_full_training_pipeline():
    """
    Test complete workflow on tiny dataset.

    This test verifies:
    1. Model loading
    2. Dataset loading and preparation
    3. Training execution
    4. Model saving
    5. Loss decrease
    """
    from llm_finetune_kit import (
        load_model,
        load_dataset,
        prepare_dataset,
        SimpleTrainer,
    )

    # Create tiny test dataset
    test_data = [
        {"prompt": "Hello", "response": "Hi there!"},
        {"prompt": "How are you?", "response": "I'm doing well, thanks!"},
        {"prompt": "Goodbye", "response": "See you later!"},
    ] * 10  # 30 examples

    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save test data
        data_path = Path(tmpdir) / "test_data.json"
        with open(data_path, "w") as f:
            json.dump(test_data, f)

        # 1. Load model (smallest model for testing)
        print("Loading model...")
        model, tokenizer, lora_config = load_model("gpt2", use_lora=True)

        assert model is not None
        assert tokenizer is not None
        assert lora_config is not None

        # 2. Load and prepare dataset
        print("Loading dataset...")
        dataset = load_dataset(str(data_path))
        assert len(dataset) == 30

        print("Preparing dataset...")
        prepared_dataset = prepare_dataset(dataset, tokenizer, max_length=128)
        assert len(prepared_dataset) == 30
        assert "input_ids" in prepared_dataset.column_names
        assert "labels" in prepared_dataset.column_names

        # 3. Create trainer
        print("Creating trainer...")
        output_dir = Path(tmpdir) / "outputs"

        trainer = SimpleTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=prepared_dataset,
            output_dir=str(output_dir),
            max_steps=10,  # Very short training
            batch_size=2,
            learning_rate=2e-4,
            logging_steps=2,
            save_steps=10,
        )

        # 4. Train (just verify it runs)
        print("Training...")
        metrics = trainer.train()

        assert "train_runtime" in metrics
        assert "train_loss" in metrics

        # 5. Verify loss decreased
        print("Checking loss...")
        assert trainer.loss_decreased(), "Loss should decrease during training"

        loss_history = trainer.get_loss_history()
        assert len(loss_history) > 0, "Should have loss history"

        print(f"Initial loss: {loss_history[0]:.4f}")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Improvement: {(loss_history[0] - loss_history[-1]):.4f}")

        # 6. Save model
        print("Saving model...")
        save_dir = Path(tmpdir) / "saved_model"
        trainer.save(str(save_dir))

        # Verify saved files
        assert (save_dir / "adapter_config.json").exists()
        assert (save_dir / "adapter_model.safetensors").exists() or (
            save_dir / "adapter_model.bin"
        ).exists()

        print("✅ Integration test passed!")


def test_model_loading():
    """Test loading different models"""
    from llm_finetune_kit import load_model, get_available_models

    # Get available models
    models = get_available_models()
    assert "gpt2" in models
    assert "mistral-7b" in models

    # Load smallest model
    model, tokenizer, config = load_model("gpt2", use_lora=True)

    assert model is not None
    assert tokenizer is not None
    assert tokenizer.pad_token is not None

    print("✅ Model loading test passed!")


def test_dataset_loading():
    """Test loading datasets"""
    from llm_finetune_kit import load_dataset, get_dataset_stats

    # Test loading from list
    data = [
        {"prompt": "Test 1", "response": "Response 1"},
        {"prompt": "Test 2", "response": "Response 2"},
    ]

    dataset = load_dataset(data)
    assert len(dataset) == 2

    # Get stats
    stats = get_dataset_stats(dataset)
    assert stats["num_examples"] == 2
    assert "prompt" in stats["columns"]
    assert "response" in stats["columns"]

    print("✅ Dataset loading test passed!")


def test_quick_trainer():
    """Test QuickTrainer convenience class"""
    from llm_finetune_kit import QuickTrainer
    import tempfile

    # Create tiny test dataset
    test_data = [
        {"prompt": f"Test {i}", "response": f"Response {i}"} for i in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save test data
        data_path = Path(tmpdir) / "test_data.json"
        with open(data_path, "w") as f:
            json.dump(test_data, f)

        # Create QuickTrainer
        output_dir = Path(tmpdir) / "quick_outputs"

        trainer = QuickTrainer(
            model_name="gpt2",
            data_source=str(data_path),
            output_dir=str(output_dir),
            max_steps=5,
            batch_size=2,
        )

        # Train
        metrics = trainer.train()
        assert "train_loss" in metrics

        print("✅ QuickTrainer test passed!")


def test_evaluation():
    """Test model evaluation"""
    from llm_finetune_kit import load_model, evaluate_model

    # Load model
    model, tokenizer, _ = load_model("gpt2", use_lora=False)

    # Evaluate on test prompts
    test_prompts = ["Hello", "How are you?"]

    results = evaluate_model(
        model, tokenizer, test_prompts, max_length=20, do_sample=False
    )

    assert len(results) == 2
    assert all("prompt" in r for r in results)
    assert all("response" in r for r in results)
    assert all("generation_time" in r for r in results)

    print("✅ Evaluation test passed!")


def test_sample_datasets():
    """Test loading sample datasets"""
    from llm_finetune_kit import load_dataset

    # Load sample datasets
    chat_dataset = load_dataset("sample:chat")
    assert len(chat_dataset) > 0
    assert "prompt" in chat_dataset.column_names
    assert "response" in chat_dataset.column_names

    instruct_dataset = load_dataset("sample:instruct")
    assert len(instruct_dataset) > 0

    print("✅ Sample datasets test passed!")


if __name__ == "__main__":
    # Run tests
    print("Running integration tests...\n")

    try:
        print("=" * 60)
        print("Test 1: Model Loading")
        print("=" * 60)
        test_model_loading()

        print("\n" + "=" * 60)
        print("Test 2: Dataset Loading")
        print("=" * 60)
        test_dataset_loading()

        print("\n" + "=" * 60)
        print("Test 3: Sample Datasets")
        print("=" * 60)
        test_sample_datasets()

        print("\n" + "=" * 60)
        print("Test 4: Evaluation")
        print("=" * 60)
        test_evaluation()

        print("\n" + "=" * 60)
        print("Test 5: QuickTrainer")
        print("=" * 60)
        test_quick_trainer()

        print("\n" + "=" * 60)
        print("Test 6: Full Training Pipeline")
        print("=" * 60)
        test_full_training_pipeline()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

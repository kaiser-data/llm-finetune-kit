"""
Command-line interface for LLM Finetune Kit.
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point for finetune command"""
    parser = argparse.ArgumentParser(
        description="LLM Finetune Kit - Fine-tune any LLM in minutes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with GPT-2
  finetune --model gpt2 --data my_data.json

  # Use sample dataset
  finetune --model gpt2 --data sample:chat --max-steps 100

  # Fine-tune Mistral 7B
  finetune --model mistral-7b --data my_data.json --max-steps 500

  # Launch web demo
  finetune --run-demo

For more information, visit: https://github.com/yourusername/llm-finetune-kit
        """,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model to fine-tune (gpt2, mistral-7b, llama-3.1-8b, phi-3-mini)",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data or sample dataset (e.g., sample:chat)",
    )

    # Training arguments
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum training steps (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (auto-configured if not specified)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-prompts",
        type=str,
        nargs="+",
        help="Prompts to evaluate model after training",
    )

    # Special modes
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Launch Gradio web interface",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs fine-tuned model",
    )
    parser.add_argument(
        "--estimate-time",
        action="store_true",
        help="Estimate training time and cost",
    )

    args = parser.parse_args()

    # Handle special modes
    if args.run_demo:
        launch_demo_ui()
        return

    if args.estimate_time:
        if not args.data:
            logger.error("❌ --data required for time estimation")
            sys.exit(1)
        estimate_training_time_cli(args)
        return

    # Validate required arguments
    if not args.data:
        logger.error("❌ --data is required")
        logger.info("💡 Try: finetune --model gpt2 --data sample:chat")
        sys.exit(1)

    # Run training
    run_training(args)


def run_training(args):
    """Run training with CLI arguments"""
    from .models import load_model
    from .datasets import load_dataset, prepare_dataset
    from .trainer import SimpleTrainer
    from .evaluate import evaluate_model, compare_models

    logger.info("🚀 LLM Finetune Kit - Starting training")
    logger.info("=" * 60)

    # Load model
    logger.info(f"📦 Loading model: {args.model}")
    model, tokenizer, _ = load_model(
        args.model,
        use_lora=True,
        lora_r=args.lora_r,
    )

    # Load base model if comparison requested
    base_model = None
    base_tokenizer = None
    if args.compare:
        logger.info("📦 Loading base model for comparison...")
        base_model, base_tokenizer, _ = load_model(args.model, use_lora=False)

    # Load dataset
    logger.info(f"📊 Loading dataset: {args.data}")
    dataset = load_dataset(args.data)

    # Prepare dataset
    logger.info("🔧 Preparing dataset...")
    prepared_dataset = prepare_dataset(dataset, tokenizer)

    # Create trainer
    trainer = SimpleTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=prepared_dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Train
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Training Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60 + "\n")

    metrics = trainer.train()

    # Save model
    trainer.save()

    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info("=" * 60 + "\n")

    # Evaluate if prompts provided
    if args.eval_prompts:
        logger.info("🧪 Evaluating model...")
        results = evaluate_model(model, tokenizer, args.eval_prompts)

        logger.info("\n📊 Evaluation Results:")
        for i, result in enumerate(results):
            logger.info(f"\n🎯 Prompt: {result['prompt']}")
            logger.info(f"💬 Response: {result['response']}")

    # Compare models if requested
    if args.compare and args.eval_prompts:
        logger.info("\n🔍 Comparing base vs fine-tuned models...")
        comparison = compare_models(
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            finetuned_model=model,
            finetuned_tokenizer=tokenizer,
            test_prompts=args.eval_prompts,
        )


def demo():
    """Launch Gradio demo interface"""
    launch_demo_ui()


def launch_demo_ui():
    """Launch Gradio web interface"""
    from .ui import launch_demo

    logger.info("🌐 Launching Gradio demo interface...")
    logger.info("💡 Access at http://localhost:7860")

    try:
        launch_demo(share=False, server_port=7860)
    except KeyboardInterrupt:
        logger.info("\n👋 Demo stopped")
    except Exception as e:
        logger.error(f"❌ Error launching demo: {e}")
        sys.exit(1)


def estimate_training_time_cli(args):
    """Estimate training time and cost"""
    from .models import estimate_training_time
    from .datasets import load_dataset

    logger.info("⏱️  Estimating training time and cost...")

    # Load dataset to get size
    dataset = load_dataset(args.data)
    dataset_size = len(dataset)

    # Estimate
    estimate = estimate_training_time(
        model_name=args.model,
        dataset_size=dataset_size,
        batch_size=args.batch_size,
        num_epochs=1,
        gpu_type="T4",
    )

    logger.info("\n" + "=" * 60)
    logger.info("📊 Training Estimate:")
    logger.info("=" * 60)
    logger.info(f"  Model: {estimate['model']} ({estimate['params']} parameters)")
    logger.info(f"  Dataset size: {estimate['dataset_size']} examples")
    logger.info(f"  Batch size: {estimate['batch_size']}")
    logger.info(f"  Epochs: {estimate['num_epochs']}")
    logger.info(f"  GPU: {estimate['gpu_type']}")
    logger.info("\n💰 Cost Estimate:")
    logger.info(f"  Estimated time: {estimate['estimated_time_minutes']} minutes")
    logger.info(f"  Estimated cost: ${estimate['estimated_cost_usd']:.2f} USD")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()

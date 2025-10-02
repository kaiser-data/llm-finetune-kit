"""
Quick evaluation and comparison tools for demo models.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import pipeline
from datasets import Dataset
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    tokenizer,
    test_prompts: List[str],
    max_length: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> List[Dict[str, str]]:
    """
    Evaluate model on test prompts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_prompts: List of test prompts
        max_length: Maximum generation length
        temperature: Sampling temperature
        do_sample: Whether to sample (vs greedy)

    Returns:
        List of dictionaries with prompt and response
    """
    logger.info(f"🧪 Evaluating model on {len(test_prompts)} prompts...")

    results = []
    model.eval()

    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            logger.info(f"  Prompt {i + 1}/{len(test_prompts)}")

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            generation_time = time.time() - start_time

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()

            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "generation_time": generation_time,
                }
            )

    logger.info("✅ Evaluation complete!")
    return results


def compare_models(
    base_model,
    base_tokenizer,
    finetuned_model,
    finetuned_tokenizer,
    test_prompts: List[str],
    max_length: int = 100,
) -> Dict[str, Any]:
    """
    Compare base model vs fine-tuned model side-by-side.

    Args:
        base_model: Base model
        base_tokenizer: Base tokenizer
        finetuned_model: Fine-tuned model
        finetuned_tokenizer: Fine-tuned tokenizer
        test_prompts: List of test prompts
        max_length: Maximum generation length

    Returns:
        Dictionary with comparison results
    """
    logger.info("🔍 Comparing base vs fine-tuned models...")

    # Evaluate base model
    logger.info("\n📊 Base Model:")
    base_results = evaluate_model(base_model, base_tokenizer, test_prompts, max_length)

    # Evaluate fine-tuned model
    logger.info("\n📊 Fine-tuned Model:")
    finetuned_results = evaluate_model(
        finetuned_model, finetuned_tokenizer, test_prompts, max_length
    )

    # Format comparison
    comparison = {
        "prompts": test_prompts,
        "base_responses": [r["response"] for r in base_results],
        "finetuned_responses": [r["response"] for r in finetuned_results],
        "base_avg_time": sum(r["generation_time"] for r in base_results)
        / len(base_results),
        "finetuned_avg_time": sum(r["generation_time"] for r in finetuned_results)
        / len(finetuned_results),
    }

    # Print comparison
    print_comparison(comparison)

    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """
    Pretty print model comparison.

    Args:
        comparison: Comparison results from compare_models
    """
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON")
    print("=" * 80 + "\n")

    for i, prompt in enumerate(comparison["prompts"]):
        print(f"🎯 Prompt {i + 1}: {prompt}")
        print("-" * 80)

        print("\n📦 Base Model:")
        print(f"  {comparison['base_responses'][i]}")

        print("\n✨ Fine-tuned Model:")
        print(f"  {comparison['finetuned_responses'][i]}")

        print("\n" + "=" * 80 + "\n")

    print(f"⏱️  Average Generation Time:")
    print(f"  Base: {comparison['base_avg_time']:.2f}s")
    print(f"  Fine-tuned: {comparison['finetuned_avg_time']:.2f}s")
    print()


def calculate_perplexity(
    model, tokenizer, test_texts: List[str], max_length: int = 512
) -> float:
    """
    Calculate average perplexity on test texts.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        test_texts: List of test texts
        max_length: Maximum sequence length

    Returns:
        Average perplexity
    """
    logger.info(f"📐 Calculating perplexity on {len(test_texts)} examples...")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in test_texts:
            # Tokenize
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])

            # Accumulate loss and token count
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"📊 Perplexity: {perplexity:.2f}")
    return perplexity


def evaluate_on_dataset(
    model,
    tokenizer,
    dataset: Dataset,
    prompt_column: str = "prompt",
    response_column: str = "response",
    n_samples: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Evaluation dataset
        prompt_column: Column name for prompts
        response_column: Column name for reference responses
        n_samples: Number of samples to evaluate

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"📊 Evaluating on {n_samples} samples from dataset...")

    # Sample dataset
    if len(dataset) > n_samples:
        indices = torch.randperm(len(dataset))[:n_samples].tolist()
        eval_data = dataset.select(indices)
    else:
        eval_data = dataset

    # Extract prompts and references
    prompts = [ex[prompt_column] for ex in eval_data]
    references = [ex[response_column] for ex in eval_data]

    # Generate responses
    results = evaluate_model(model, tokenizer, prompts)

    # Calculate metrics
    generated_responses = [r["response"] for r in results]
    avg_gen_time = sum(r["generation_time"] for r in results) / len(results)

    # Simple metrics
    avg_gen_length = sum(len(r.split()) for r in generated_responses) / len(
        generated_responses
    )
    avg_ref_length = sum(len(r.split()) for r in references) / len(references)

    metrics = {
        "n_samples": n_samples,
        "avg_generation_time": avg_gen_time,
        "avg_generated_length": avg_gen_length,
        "avg_reference_length": avg_ref_length,
        "samples": [
            {
                "prompt": p,
                "generated": g,
                "reference": r,
            }
            for p, g, r in zip(prompts, generated_responses, references)
        ],
    }

    logger.info(f"✅ Evaluation complete!")
    logger.info(f"  Avg generation time: {avg_gen_time:.2f}s")
    logger.info(f"  Avg generated length: {avg_gen_length:.1f} words")
    logger.info(f"  Avg reference length: {avg_ref_length:.1f} words")

    return metrics


def quick_test(model, tokenizer, prompt: str, max_length: int = 100) -> str:
    """
    Quick test of model on a single prompt.

    Args:
        model: Model to test
        tokenizer: Tokenizer
        prompt: Test prompt
        max_length: Maximum generation length

    Returns:
        Generated response
    """
    results = evaluate_model(model, tokenizer, [prompt], max_length)
    response = results[0]["response"]

    print(f"\n🎯 Prompt: {prompt}")
    print(f"💬 Response: {response}\n")

    return response


def benchmark_speed(
    model, tokenizer, n_iterations: int = 10, prompt_length: int = 50
) -> Dict[str, float]:
    """
    Benchmark model generation speed.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer
        n_iterations: Number of iterations
        prompt_length: Length of test prompt in tokens

    Returns:
        Dictionary with speed metrics
    """
    logger.info(f"⏱️  Benchmarking speed over {n_iterations} iterations...")

    # Create test prompt
    test_prompt = " ".join(["test"] * prompt_length)

    times = []
    model.eval()

    with torch.no_grad():
        for i in range(n_iterations):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.time() - start_time
            times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    metrics = {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "tokens_per_second": 50 / avg_time,
    }

    logger.info(f"✅ Benchmark complete!")
    logger.info(f"  Avg time: {avg_time:.3f}s")
    logger.info(f"  Tokens/sec: {metrics['tokens_per_second']:.1f}")

    return metrics

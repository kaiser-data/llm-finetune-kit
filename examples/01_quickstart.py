"""
Quickstart Example - Fine-tune GPT-2 in 5 minutes

This example demonstrates the complete workflow:
1. Load model
2. Load dataset
3. Train
4. Evaluate
5. Compare base vs fine-tuned
"""

print("=" * 60)
print("LLM Finetune Kit - Quickstart Example")
print("=" * 60 + "\n")

# ============================================================================
# Method 1: Ultra-Quick (3 lines)
# ============================================================================

print("📚 Method 1: Ultra-Quick Training (3 lines)\n")

from llm_finetune_kit import QuickTrainer

# This is all you need!
trainer = QuickTrainer("gpt2", "sample:chat", max_steps=50)
trainer.train()
trainer.save()

print("\n✅ Ultra-quick training complete!")

# ============================================================================
# Method 2: Step-by-Step (more control)
# ============================================================================

print("\n" + "=" * 60)
print("📚 Method 2: Step-by-Step Training")
print("=" * 60 + "\n")

from llm_finetune_kit import (
    load_model,
    load_dataset,
    prepare_dataset,
    SimpleTrainer,
    evaluate_model,
    compare_models,
)

# Step 1: Load model
print("1️⃣ Loading model...")
model, tokenizer, lora_config = load_model("gpt2", use_lora=True)
print("   ✓ Model loaded\n")

# Load base model for comparison
print("   Loading base model for comparison...")
base_model, base_tokenizer, _ = load_model("gpt2", use_lora=False)
print("   ✓ Base model loaded\n")

# Step 2: Load dataset
print("2️⃣ Loading dataset...")
dataset = load_dataset("sample:chat")
print(f"   ✓ Loaded {len(dataset)} examples\n")

# Step 3: Prepare dataset
print("3️⃣ Preparing dataset...")
prepared_dataset = prepare_dataset(dataset, tokenizer, max_length=256)
print(f"   ✓ Prepared {len(prepared_dataset)} examples\n")

# Step 4: Create trainer
print("4️⃣ Creating trainer...")
trainer = SimpleTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=prepared_dataset,
    output_dir="./quickstart_outputs",
    max_steps=100,
    batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
)
print("   ✓ Trainer created\n")

# Step 5: Train
print("5️⃣ Training model...")
print("=" * 60)
metrics = trainer.train()
print("=" * 60)
print("   ✓ Training complete!\n")

# Print metrics
print("📊 Training Metrics:")
print(f"   Loss: {metrics['train_loss']:.4f}")
print(f"   Runtime: {metrics['train_runtime']:.1f}s")
print(f"   Samples/sec: {metrics['train_samples_per_second']:.2f}\n")

# Step 6: Save model
print("6️⃣ Saving model...")
trainer.save("./quickstart_outputs/final_model")
print("   ✓ Model saved\n")

# Step 7: Evaluate
print("7️⃣ Evaluating model...")
print("=" * 60)

test_prompts = [
    "How do I reset my password?",
    "What are your business hours?",
    "How can I track my order?",
]

print("\n🧪 Testing Fine-tuned Model:\n")
for prompt in test_prompts:
    results = evaluate_model(model, tokenizer, [prompt], max_length=150)
    print(f"Prompt: {prompt}")
    print(f"Response: {results[0]['response']}\n")

# Step 8: Compare base vs fine-tuned
print("\n8️⃣ Comparing base vs fine-tuned models...")
print("=" * 60 + "\n")

comparison = compare_models(
    base_model=base_model,
    base_tokenizer=base_tokenizer,
    finetuned_model=model,
    finetuned_tokenizer=tokenizer,
    test_prompts=test_prompts,
)

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("🎉 Quickstart Complete!")
print("=" * 60)
print("\n📁 Your model is saved at: ./quickstart_outputs/final_model")
print("\n💡 Next steps:")
print("   1. Try with your own dataset")
print("   2. Experiment with different models (mistral-7b, llama-3.1-8b)")
print("   3. Launch the web demo: finetune-demo")
print("   4. Check out more examples in the examples/ directory")
print("\n📚 Documentation: https://github.com/kaiser-data/llm-finetune-kit")
print("=" * 60 + "\n")

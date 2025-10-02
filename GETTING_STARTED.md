# Getting Started with LLM Finetune Kit

Welcome! This guide will help you fine-tune your first LLM in under 5 minutes.

## 🚀 Quickest Way to Get Started

### Option 1: 3-Line Code (Fastest)

```python
from llm_finetune_kit import QuickTrainer

trainer = QuickTrainer("gpt2", "sample:chat")
trainer.train()
```

**Done!** Your model is now fine-tuned on customer support conversations.

### Option 2: Web Interface (No Code)

```bash
finetune-demo
```

Then open http://localhost:7860 in your browser and follow the UI.

### Option 3: Command Line

```bash
finetune --model gpt2 --data sample:chat --max-steps 100
```

## 📚 Your First Training Session

Let's walk through a complete example:

### Step 1: Install

```bash
pip install llm-finetune-kit
```

### Step 2: Create Your Script

Create `my_first_finetune.py`:

```python
from llm_finetune_kit import load_model, load_dataset, prepare_dataset, SimpleTrainer

# 1. Load GPT-2 (smallest, fastest model)
print("Loading model...")
model, tokenizer, _ = load_model("gpt2")

# 2. Load sample dataset (customer support)
print("Loading dataset...")
dataset = load_dataset("sample:chat")
prepared = prepare_dataset(dataset, tokenizer)

# 3. Train for 100 steps (~5 minutes on GPU)
print("Training...")
trainer = SimpleTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=prepared,
    max_steps=100,
    output_dir="./my_first_model"
)

metrics = trainer.train()
trainer.save()

print(f"✅ Training complete! Loss: {metrics['train_loss']:.4f}")
```

### Step 3: Run It

```bash
python my_first_finetune.py
```

### Step 4: Test Your Model

```python
from llm_finetune_kit import evaluate_model

# Load your fine-tuned model
model, tokenizer, _ = load_model("gpt2")

# Test it
results = evaluate_model(
    model, 
    tokenizer, 
    ["How do I reset my password?"]
)

print(results[0]['response'])
```

## 🎯 What to Try Next

### 1. Use Your Own Data

Create `my_data.json`:

```json
[
  {"prompt": "What is Python?", "response": "Python is a programming language..."},
  {"prompt": "Explain functions", "response": "Functions are reusable blocks of code..."}
]
```

Then train:

```python
from llm_finetune_kit import QuickTrainer

trainer = QuickTrainer("gpt2", "my_data.json", max_steps=200)
trainer.train()
```

### 2. Try a Larger Model

```python
# Mistral 7B (requires 12GB+ GPU)
trainer = QuickTrainer("mistral-7b", "sample:chat", max_steps=500)
trainer.train()
```

### 3. Compare Models

```python
from llm_finetune_kit import load_model, compare_models

base_model, base_tok, _ = load_model("gpt2", use_lora=False)
ft_model, ft_tok, _ = load_model("gpt2")

compare_models(
    base_model, base_tok,
    ft_model, ft_tok,
    ["How do I reset my password?"]
)
```

## 💡 Common Use Cases

### Chatbot Assistant

```python
from llm_finetune_kit import QuickTrainer

# Train on customer support conversations
trainer = QuickTrainer("gpt2", "sample:chat", max_steps=200)
trainer.train()
```

### Educational Tutor

```python
# Train on educational Q&A
trainer = QuickTrainer("gpt2", "sample:instruct", max_steps=200)
trainer.train()
```

### Custom Domain Expert

```python
# Train on your domain-specific data
trainer = QuickTrainer("mistral-7b", "my_medical_data.json")
trainer.train()
```

## 🔧 Common Configurations

### Fast Demo (5 minutes)

```python
SimpleTrainer(
    model, tokenizer, dataset,
    max_steps=100,
    batch_size=4,
    output_dir="./quick_demo"
)
```

### Standard Training (15 minutes)

```python
SimpleTrainer(
    model, tokenizer, dataset,
    max_steps=500,
    batch_size=4,
    learning_rate=2e-4,
    output_dir="./standard"
)
```

### High Quality (30+ minutes)

```python
SimpleTrainer(
    model, tokenizer, dataset,
    max_steps=1000,
    num_epochs=3,
    batch_size=8,
    eval_dataset=val_dataset,
    output_dir="./high_quality"
)
```

## ⚠️ Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
trainer = SimpleTrainer(
    model, tokenizer, dataset,
    batch_size=1,  # Reduce from 4
    gradient_accumulation_steps=4  # Maintain effective batch size
)
```

### Training Too Slow

```python
# Use smaller model
model, tokenizer, _ = load_model("gpt2")  # Instead of mistral-7b

# Reduce steps
trainer = SimpleTrainer(
    model, tokenizer, dataset,
    max_steps=50  # Instead of 500
)
```

### Poor Quality Results

```python
# Train longer
trainer = SimpleTrainer(
    model, tokenizer, dataset,
    max_steps=1000,  # More steps
    num_epochs=3  # Multiple epochs
)

# Use more/better data
dataset = load_dataset("my_large_dataset.json")  # 1000+ examples
```

## 📖 Learn More

- **Full Documentation**: See `README.md`
- **API Reference**: See `PROJECT_SUMMARY.md`
- **Installation Guide**: See `INSTALL.md`
- **Examples**: Check `examples/` directory
- **Support**: GitHub Issues or Discord

## 🎉 Success!

You've fine-tuned your first LLM! Here's what you learned:

✅ Load models with smart defaults  
✅ Prepare datasets for training  
✅ Train with simple configuration  
✅ Evaluate model outputs  
✅ Compare base vs fine-tuned models  

**Next Steps:**
1. Try with your own dataset
2. Experiment with different models
3. Share your results!

---

*Happy fine-tuning! 🚀*

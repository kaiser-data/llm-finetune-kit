# Installation Guide

## Quick Install

### From PyPI (Coming Soon)

```bash
pip install llm-finetune-kit
```

### From Source (Current Method)

```bash
# Clone repository
git clone https://github.com/kaiser-data/llm-finetune-kit.git
cd llm-finetune-kit

# Install in development mode
pip install -e .
```

## Requirements

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB for models and dependencies

### GPU Requirements by Model

| Model | Min VRAM | Recommended | Quantization |
|-------|----------|-------------|--------------|
| GPT-2 | 4GB | 8GB | Optional |
| GPT-2 Medium | 6GB | 12GB | Optional |
| Phi-3 Mini | 8GB | 16GB | Optional |
| Mistral 7B | 12GB | 24GB | Required (4-bit) |
| Llama 3.1 8B | 16GB | 24GB | Required (4-bit) |

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv llm-env
source llm-env/bin/activate  # On Windows: llm-env\Scripts\activate

# Or using conda
conda create -n llm-env python=3.10
conda activate llm-env
```

### 2. Install PyTorch

Install PyTorch with CUDA support (for GPU):

```bash
# CUDA 11.8
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended for training)
pip install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cpu
```

Visit [PyTorch.org](https://pytorch.org/get-started/locally/) for other configurations.

### 3. Install LLM Finetune Kit

```bash
# From source
git clone https://github.com/kaiser-data/llm-finetune-kit.git
cd llm-finetune-kit
pip install -e .

# This will install all dependencies from requirements.txt
```

### 4. Verify Installation

```bash
# Check if installed correctly
python -c "from llm_finetune_kit import load_model; print('✅ Installation successful!')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run quick test
python -c "from llm_finetune_kit import QuickTrainer; print('✅ All imports working!')"
```

## Google Colab Installation

```python
# In a Colab notebook
!pip install git+https://github.com/kaiser-data/llm-finetune-kit.git

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}")

# Quick test
from llm_finetune_kit import QuickTrainer
trainer = QuickTrainer("gpt2", "sample:chat", max_steps=10)
trainer.train()
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
- Use smaller model (gpt2 instead of mistral-7b)
- Reduce batch size: `--batch-size 1`
- Enable quantization for large models
- Use gradient accumulation: `gradient_accumulation_steps=4`

### Issue: `bitsandbytes` Not Found

**Solution:**
```bash
# Linux
pip install bitsandbytes>=0.44.0

# Windows (requires pre-compiled wheels)
pip install bitsandbytes-windows
```

### Issue: Slow Training on CPU

**Solution:**
- Install CUDA-enabled PyTorch
- Use Google Colab with free T4 GPU
- Consider cloud GPU providers (RunPod, Lambda Labs)

### Issue: Import Errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+

# Verify installation
pip show llm-finetune-kit
```

### Issue: Gradio Not Launching

**Solution:**
```bash
# Install gradio separately
pip install gradio>=4.0.0

# Check port availability
finetune-demo --server-port 7861
```

## Development Installation

For contributing to the project:

```bash
# Clone repository
git clone https://github.com/kaiser-data/llm-finetune-kit.git
cd llm-finetune-kit

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
pytest tests/
```

## Upgrading

### From Source

```bash
cd llm-finetune-kit
git pull origin main
pip install -e . --upgrade
```

### From PyPI (When Available)

```bash
pip install llm-finetune-kit --upgrade
```

## Uninstallation

```bash
pip uninstall llm-finetune-kit
```

## Platform-Specific Notes

### Windows

- Install Visual Studio Build Tools for bitsandbytes
- Use Windows Subsystem for Linux (WSL2) for better compatibility
- May need to install CUDA Toolkit manually

### macOS

- MPS backend supported for Apple Silicon (M1/M2)
- Limited quantization support (no bitsandbytes)
- Recommended: Use smaller models (GPT-2, Phi-3 Mini)

### Linux

- Best compatibility for all features
- NVIDIA CUDA required for GPU acceleration
- Recommended platform for development

## Cloud Platforms

### Google Colab

```python
# Free T4 GPU
!pip install git+https://github.com/kaiser-data/llm-finetune-kit.git
```

### Kaggle Notebooks

```python
# Free P100 or T4 GPU
!pip install git+https://github.com/kaiser-data/llm-finetune-kit.git
```

### AWS SageMaker

```python
# Configure instance with GPU
!pip install git+https://github.com/kaiser-data/llm-finetune-kit.git
```

## Next Steps

After installation:

1. **Quick Test**: `python examples/01_quickstart.py`
2. **Launch Demo**: `finetune-demo`
3. **Read Docs**: See `README.md` for tutorials
4. **Join Community**: [Discord](#) | [GitHub Discussions](#)

## Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/kaiser-data/llm-finetune-kit/issues)
- 💬 Discord: [Join Server](#)
- 📚 Docs: [Full Documentation](#)

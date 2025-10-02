"""
Model loading with smart defaults for demo-focused fine-tuning.

Automatically configures quantization, LoRA, and memory optimization
based on available GPU resources.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from typing import Optional, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model registry with metadata
AVAILABLE_MODELS = {
    "gpt2": {
        "model_id": "gpt2",
        "params": "124M",
        "min_vram": 4,
        "recommended_vram": 8,
        "default_lora_r": 8,
        "context_length": 1024,
    },
    "gpt2-medium": {
        "model_id": "gpt2-medium",
        "params": "355M",
        "min_vram": 6,
        "recommended_vram": 12,
        "default_lora_r": 16,
        "context_length": 1024,
    },
    "gpt2-large": {
        "model_id": "gpt2-large",
        "params": "774M",
        "min_vram": 8,
        "recommended_vram": 16,
        "default_lora_r": 16,
        "context_length": 1024,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "params": "7B",
        "min_vram": 12,
        "recommended_vram": 24,
        "default_lora_r": 16,
        "context_length": 4096,
        "requires_quantization": True,
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "params": "8B",
        "min_vram": 16,
        "recommended_vram": 24,
        "default_lora_r": 16,
        "context_length": 8192,
        "requires_quantization": True,
    },
    "phi-3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "params": "3.8B",
        "min_vram": 8,
        "recommended_vram": 16,
        "default_lora_r": 16,
        "context_length": 4096,
    },
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get registry of available models with their specifications.

    Returns:
        Dictionary mapping model names to their metadata
    """
    return AVAILABLE_MODELS.copy()


def get_gpu_memory() -> float:
    """
    Get available GPU memory in GB.

    Returns:
        Available GPU memory in GB, or 0 if no GPU available
    """
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 Total GPU memory: {gpu_memory:.2f} GB")
        return gpu_memory
    else:
        logger.warning("⚠️  No GPU detected. Training will be slow on CPU.")
        return 0


def auto_configure_quantization(
    model_name: str, gpu_memory: float, force_quantization: Optional[bool] = None
) -> Optional[BitsAndBytesConfig]:
    """
    Automatically configure quantization based on model size and GPU memory.

    Args:
        model_name: Name of the model from AVAILABLE_MODELS
        gpu_memory: Available GPU memory in GB
        force_quantization: Override automatic decision (True/False/None)

    Returns:
        BitsAndBytesConfig if quantization needed, None otherwise
    """
    model_info = AVAILABLE_MODELS.get(model_name, {})
    recommended_vram = model_info.get("recommended_vram", 16)
    requires_quantization = model_info.get("requires_quantization", False)

    # Decision logic
    use_quantization = force_quantization
    if use_quantization is None:
        use_quantization = requires_quantization or gpu_memory < recommended_vram

    if not use_quantization:
        logger.info("✅ Loading model in full precision")
        return None

    # Configure 4-bit quantization for memory efficiency
    logger.info("⚙️  Configuring 4-bit quantization for memory efficiency")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    return bnb_config


def get_lora_config(
    model_name: str,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
) -> LoraConfig:
    """
    Get LoRA configuration with smart defaults.

    Args:
        model_name: Name of the model from AVAILABLE_MODELS
        lora_r: LoRA rank (auto-configured if None)
        lora_alpha: LoRA alpha (defaults to 2*lora_r)
        lora_dropout: LoRA dropout probability
        target_modules: Modules to apply LoRA to (auto-detected if None)

    Returns:
        Configured LoraConfig
    """
    model_info = AVAILABLE_MODELS.get(model_name, {})

    # Auto-configure LoRA rank
    if lora_r is None:
        lora_r = model_info.get("default_lora_r", 16)

    if lora_alpha is None:
        lora_alpha = lora_r * 2

    # Auto-detect target modules based on model architecture
    if target_modules is None:
        model_id = model_info.get("model_id", "")
        if "gpt2" in model_id.lower():
            target_modules = ["c_attn", "c_proj"]
        elif "mistral" in model_id.lower() or "llama" in model_id.lower():
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "phi" in model_id.lower():
            target_modules = ["qkv_proj", "o_proj"]
        else:
            # Generic fallback
            target_modules = ["q_proj", "v_proj"]

    logger.info(f"🎯 LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"🎯 Target modules: {target_modules}")

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model(
    model_name: str,
    use_lora: bool = True,
    quantization: Optional[bool] = None,
    lora_r: Optional[int] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> Tuple[Any, Any, Optional[LoraConfig]]:
    """
    Load model with smart defaults for demo-focused fine-tuning.

    Automatically configures:
    - Quantization based on GPU memory
    - LoRA parameters based on model size
    - Memory optimization settings

    Args:
        model_name: Model name from AVAILABLE_MODELS or HuggingFace model ID
        use_lora: Whether to apply LoRA (recommended for fine-tuning)
        quantization: Force quantization on/off (None = auto-detect)
        lora_r: LoRA rank override
        device_map: Device placement strategy
        trust_remote_code: Trust remote code (needed for some models)

    Returns:
        Tuple of (model, tokenizer, lora_config)

    Example:
        >>> model, tokenizer, lora_config = load_model("gpt2")
        >>> # Ready for training!
    """
    # Validate model
    if model_name not in AVAILABLE_MODELS:
        logger.warning(f"⚠️  '{model_name}' not in registry. Using as HuggingFace model ID.")
        model_id = model_name
        model_info = {}
    else:
        model_info = AVAILABLE_MODELS[model_name]
        model_id = model_info["model_id"]
        logger.info(f"📦 Loading {model_name} ({model_info.get('params', 'unknown')} parameters)")

    # Check GPU and configure quantization
    gpu_memory = get_gpu_memory()

    # Show recommendations
    min_vram = model_info.get("min_vram", 8)
    recommended_vram = model_info.get("recommended_vram", 16)

    if gpu_memory > 0 and gpu_memory < min_vram:
        logger.warning(f"⚠️  GPU memory ({gpu_memory:.1f}GB) below minimum ({min_vram}GB)")
        logger.warning(f"💡 Consider using a smaller model like 'gpt2' or 'phi-3-mini'")

    bnb_config = auto_configure_quantization(model_name, gpu_memory, quantization)

    # Load tokenizer
    logger.info("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("⚙️  Set pad_token = eos_token")

    # Load model
    logger.info("🔧 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    # Apply LoRA if requested
    lora_config = None
    if use_lora:
        logger.info("🎨 Preparing model for LoRA fine-tuning...")

        # Prepare for k-bit training if quantized
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)

        # Get LoRA config and apply
        lora_config = get_lora_config(model_name, lora_r=lora_r)
        model = get_peft_model(model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✨ Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    logger.info("✅ Model loaded successfully!")

    return model, tokenizer, lora_config


def save_model(model, tokenizer, output_dir: str):
    """
    Save fine-tuned model and tokenizer.

    Args:
        model: Fine-tuned model (with LoRA adapters)
        tokenizer: Tokenizer
        output_dir: Directory to save to
    """
    logger.info(f"💾 Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("✅ Model saved successfully!")


def estimate_training_time(
    model_name: str,
    dataset_size: int,
    batch_size: int = 1,
    num_epochs: int = 1,
    gpu_type: str = "T4",
) -> Dict[str, Any]:
    """
    Estimate training time and cost for a given configuration.

    Args:
        model_name: Model name from AVAILABLE_MODELS
        dataset_size: Number of training examples
        batch_size: Training batch size
        num_epochs: Number of training epochs
        gpu_type: GPU type (T4, V100, A100, etc.)

    Returns:
        Dictionary with time and cost estimates
    """
    model_info = AVAILABLE_MODELS.get(model_name, {})
    params = model_info.get("params", "unknown")

    # Rough time estimates (seconds per example)
    time_per_example = {
        "gpt2": {"T4": 0.5, "V100": 0.3, "A100": 0.15},
        "gpt2-medium": {"T4": 1.0, "V100": 0.6, "A100": 0.3},
        "gpt2-large": {"T4": 2.0, "V100": 1.2, "A100": 0.6},
        "mistral-7b": {"T4": 3.0, "V100": 1.8, "A100": 0.9},
        "llama-3.1-8b": {"T4": 3.5, "V100": 2.0, "A100": 1.0},
        "phi-3-mini": {"T4": 2.0, "V100": 1.2, "A100": 0.6},
    }

    base_time = time_per_example.get(model_name, {}).get(gpu_type, 1.0)
    total_time_seconds = base_time * dataset_size * num_epochs / batch_size
    total_time_minutes = total_time_seconds / 60

    # Cost estimates ($/hour)
    gpu_costs = {
        "T4": 0.35,
        "V100": 0.80,
        "A100": 2.50,
    }

    cost_per_hour = gpu_costs.get(gpu_type, 0.50)
    total_cost = (total_time_minutes / 60) * cost_per_hour

    return {
        "estimated_time_minutes": round(total_time_minutes, 1),
        "estimated_cost_usd": round(total_cost, 2),
        "gpu_type": gpu_type,
        "model": model_name,
        "params": params,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }

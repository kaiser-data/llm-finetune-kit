"""
Gradio-based demo interface for interactive fine-tuning.

Provides web UI for:
- Uploading datasets
- Configuring training
- Monitoring progress
- Testing models before/after training
"""

import gradio as gr
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import time

from .models import load_model, get_available_models, estimate_training_time
from .datasets import load_dataset, prepare_dataset, get_dataset_stats
from .trainer import SimpleTrainer
from .evaluate import evaluate_model, compare_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingSession:
    """Manages a training session with progress tracking"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.base_model = None
        self.base_tokenizer = None
        self.is_training = False
        self.training_complete = False
        self.progress = 0
        self.status = "Ready"

    def reset(self):
        """Reset session"""
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.is_training = False
        self.training_complete = False
        self.progress = 0
        self.status = "Ready"


# Global session
session = TrainingSession()


def load_model_ui(model_name: str) -> str:
    """Load model from UI"""
    try:
        session.status = f"Loading {model_name}..."
        logger.info(f"Loading model: {model_name}")

        # Load model
        session.model, session.tokenizer, _ = load_model(model_name, use_lora=True)

        # Also load base model for comparison
        session.base_model, session.base_tokenizer, _ = load_model(
            model_name, use_lora=False
        )

        session.status = "Model loaded ✅"
        return f"✅ {model_name} loaded successfully!"

    except Exception as e:
        session.status = "Error loading model ❌"
        return f"❌ Error: {str(e)}"


def load_dataset_ui(dataset_file: Optional[Any], sample_dataset: str) -> str:
    """Load dataset from UI"""
    try:
        session.status = "Loading dataset..."

        if dataset_file is not None:
            # User uploaded file
            file_path = dataset_file.name
            logger.info(f"Loading uploaded file: {file_path}")
            dataset = load_dataset(file_path)
        else:
            # Use sample dataset
            logger.info(f"Loading sample dataset: {sample_dataset}")
            dataset = load_dataset(f"sample:{sample_dataset}")

        # Get stats
        stats = get_dataset_stats(dataset)

        session.dataset = dataset
        session.status = "Dataset loaded ✅"

        return f"""✅ Dataset loaded!

📊 Statistics:
- Examples: {stats['num_examples']}
- Columns: {', '.join(stats['columns'])}
"""

    except Exception as e:
        session.status = "Error loading dataset ❌"
        return f"❌ Error: {str(e)}"


def start_training_ui(
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    output_dir: str,
) -> str:
    """Start training from UI"""
    try:
        if session.model is None:
            return "❌ Please load a model first"

        if not hasattr(session, "dataset"):
            return "❌ Please load a dataset first"

        session.status = "Preparing training..."
        session.is_training = True
        session.training_complete = False

        # Prepare dataset
        prepared_dataset = prepare_dataset(session.dataset, session.tokenizer)

        # Create trainer
        session.trainer = SimpleTrainer(
            model=session.model,
            tokenizer=session.tokenizer,
            train_dataset=prepared_dataset,
            output_dir=output_dir,
            max_steps=max_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=max(1, max_steps // 10),
            save_steps=max_steps,  # Save only at end
        )

        # Train in background thread
        def train_thread():
            try:
                session.status = "Training... 🚀"
                session.trainer.train()
                session.training_complete = True
                session.status = "Training complete ✅"
            except Exception as e:
                session.status = f"Training error: {str(e)} ❌"
                logger.error(f"Training error: {e}")
            finally:
                session.is_training = False

        thread = threading.Thread(target=train_thread)
        thread.start()

        return f"""🚀 Training started!

Configuration:
- Max steps: {max_steps}
- Batch size: {batch_size}
- Learning rate: {learning_rate}
- Output: {output_dir}

Check the 'Status' field for progress updates.
"""

    except Exception as e:
        session.is_training = False
        session.status = "Error ❌"
        return f"❌ Error: {str(e)}"


def get_training_status() -> str:
    """Get current training status"""
    return session.status


def test_model_ui(prompt: str, use_finetuned: bool = False) -> str:
    """Test model from UI"""
    try:
        if session.model is None:
            return "❌ Please load a model first"

        model = session.model if use_finetuned else session.base_model
        tokenizer = session.tokenizer if use_finetuned else session.base_tokenizer

        if model is None:
            return "❌ Model not available"

        session.status = "Generating response..."

        results = evaluate_model(model, tokenizer, [prompt], max_length=150)
        response = results[0]["response"]
        gen_time = results[0]["generation_time"]

        session.status = "Ready"

        model_type = "Fine-tuned" if use_finetuned else "Base"
        return f"""💬 {model_type} Model Response:

{response}

⏱️ Generation time: {gen_time:.2f}s
"""

    except Exception as e:
        session.status = "Error ❌"
        return f"❌ Error: {str(e)}"


def compare_models_ui(prompts_text: str) -> str:
    """Compare base vs fine-tuned models"""
    try:
        if session.model is None or session.base_model is None:
            return "❌ Please load models first"

        if not session.training_complete:
            return "⚠️ Training not complete. Using current model state."

        # Parse prompts
        prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]

        if not prompts:
            return "❌ Please enter at least one prompt"

        session.status = "Comparing models..."

        # Compare
        comparison = compare_models(
            base_model=session.base_model,
            base_tokenizer=session.base_tokenizer,
            finetuned_model=session.model,
            finetuned_tokenizer=session.tokenizer,
            test_prompts=prompts,
        )

        # Format results
        result = "📊 MODEL COMPARISON\n\n"

        for i, prompt in enumerate(prompts):
            result += f"🎯 Prompt: {prompt}\n\n"
            result += f"📦 Base Model:\n{comparison['base_responses'][i]}\n\n"
            result += f"✨ Fine-tuned Model:\n{comparison['finetuned_responses'][i]}\n\n"
            result += "-" * 80 + "\n\n"

        session.status = "Ready"
        return result

    except Exception as e:
        session.status = "Error ❌"
        return f"❌ Error: {str(e)}"


def create_demo_ui() -> gr.Blocks:
    """Create Gradio demo interface"""

    with gr.Blocks(
        title="LLM Finetune Kit - Interactive Demo", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
        # 🚀 LLM Finetune Kit - Interactive Demo

        Fine-tune any LLM in minutes with this easy-to-use interface!

        **Steps:**
        1. Select and load a model
        2. Upload or select a dataset
        3. Configure and start training
        4. Test and compare results
        """
        )

        # Status bar
        status_box = gr.Textbox(
            label="Status", value="Ready", interactive=False, every=1
        )
        status_box.value = get_training_status

        with gr.Tab("1️⃣ Model Selection"):
            gr.Markdown("### Select a model to fine-tune")

            model_dropdown = gr.Dropdown(
                choices=list(get_available_models().keys()),
                value="gpt2",
                label="Model",
                info="Start with GPT-2 for fastest results",
            )

            model_info = gr.Markdown(
                """
            **GPT-2** (124M parameters)
            - Min VRAM: 4GB
            - Recommended VRAM: 8GB
            - Training time: ~5 minutes (100 steps)
            """
            )

            load_model_btn = gr.Button("Load Model", variant="primary")
            model_output = gr.Textbox(label="Result", lines=3)

            load_model_btn.click(
                fn=load_model_ui, inputs=[model_dropdown], outputs=[model_output]
            )

        with gr.Tab("2️⃣ Dataset"):
            gr.Markdown("### Load your training data")

            with gr.Row():
                with gr.Column():
                    dataset_file = gr.File(
                        label="Upload Dataset (JSON)",
                        file_types=[".json"],
                        file_count="single",
                    )

                with gr.Column():
                    sample_dataset = gr.Radio(
                        choices=["chat", "instruct"],
                        value="chat",
                        label="Or use sample dataset",
                    )

            load_dataset_btn = gr.Button("Load Dataset", variant="primary")
            dataset_output = gr.Textbox(label="Result", lines=5)

            load_dataset_btn.click(
                fn=load_dataset_ui,
                inputs=[dataset_file, sample_dataset],
                outputs=[dataset_output],
            )

        with gr.Tab("3️⃣ Training"):
            gr.Markdown("### Configure and start training")

            with gr.Row():
                max_steps = gr.Slider(
                    minimum=10,
                    maximum=1000,
                    value=100,
                    step=10,
                    label="Max Steps",
                    info="More steps = better results but longer training",
                )

                batch_size = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Batch Size",
                    info="Reduce if you get OOM errors",
                )

            with gr.Row():
                learning_rate = gr.Number(
                    value=2e-4,
                    label="Learning Rate",
                    info="Default: 2e-4",
                )

                output_dir = gr.Textbox(
                    value="./demo_outputs",
                    label="Output Directory",
                )

            train_btn = gr.Button("Start Training 🚀", variant="primary", size="lg")
            training_output = gr.Textbox(label="Training Info", lines=8)

            train_btn.click(
                fn=start_training_ui,
                inputs=[max_steps, batch_size, learning_rate, output_dir],
                outputs=[training_output],
            )

        with gr.Tab("4️⃣ Test & Compare"):
            gr.Markdown("### Test your fine-tuned model")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Single Prompt Test**")
                    test_prompt = gr.Textbox(
                        label="Test Prompt",
                        placeholder="Enter a prompt to test...",
                        lines=3,
                    )

                    with gr.Row():
                        test_base_btn = gr.Button("Test Base Model")
                        test_ft_btn = gr.Button("Test Fine-tuned Model", variant="primary")

                    test_output = gr.Textbox(label="Response", lines=6)

                    test_base_btn.click(
                        fn=lambda p: test_model_ui(p, use_finetuned=False),
                        inputs=[test_prompt],
                        outputs=[test_output],
                    )

                    test_ft_btn.click(
                        fn=lambda p: test_model_ui(p, use_finetuned=True),
                        inputs=[test_prompt],
                        outputs=[test_output],
                    )

                with gr.Column():
                    gr.Markdown("**Side-by-Side Comparison**")
                    compare_prompts = gr.Textbox(
                        label="Prompts (one per line)",
                        placeholder="Enter prompts to compare...\nOne per line",
                        lines=5,
                    )

                    compare_btn = gr.Button("Compare Models", variant="primary")
                    compare_output = gr.Textbox(label="Comparison", lines=15)

                    compare_btn.click(
                        fn=compare_models_ui,
                        inputs=[compare_prompts],
                        outputs=[compare_output],
                    )

        gr.Markdown(
            """
        ---

        ### 💡 Tips
        - Start with GPT-2 and sample datasets for fastest results
        - Use 100-500 steps for quick demos
        - Reduce batch size if you get out-of-memory errors
        - Training on Google Colab T4 GPU: ~5-15 minutes

        ### 📚 Resources
        - [Documentation](#)
        - [GitHub Repository](#)
        - [Examples & Tutorials](#)
        """
        )

    return demo


def launch_demo(share: bool = False, server_port: int = 7860):
    """
    Launch Gradio demo interface.

    Args:
        share: Create public share link
        server_port: Port to run server on
    """
    demo = create_demo_ui()

    logger.info(f"🌐 Launching demo on port {server_port}")
    logger.info(f"📱 Share link: {'Enabled' if share else 'Disabled'}")

    demo.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0",
        show_error=True,
    )


if __name__ == "__main__":
    launch_demo()

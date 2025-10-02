"""
Setup script for LLM Finetune Kit
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="llm-finetune-kit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Beginner-friendly Python library for fine-tuning small-to-medium LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-finetune-kit",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-finetune-kit/issues",
        "Documentation": "https://github.com/yourusername/llm-finetune-kit#readme",
        "Source": "https://github.com/yourusername/llm-finetune-kit",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finetune=llm_finetune_kit.cli:main",
            "finetune-demo=llm_finetune_kit.cli:demo",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_finetune_kit": [
            "../configs/*.yaml",
            "../datasets/*.json",
        ],
    },
    keywords="llm fine-tuning machine-learning transformers lora peft gpt llama mistral",
    zip_safe=False,
)

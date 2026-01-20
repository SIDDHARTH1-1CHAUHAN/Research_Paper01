"""
Multi-Agent Fact-Checking Pipeline
A research-grade fact-checking system using specialized agents
Based on "Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="multi-agent-fact-checker",
    version="0.1.0",
    author="Research Team",
    author_email="",
    description="A multi-agent system for automated fact-checking with explainable AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "ollama>=0.3.0",
        "duckduckgo-search>=6.0.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "tenacity>=8.2.0",
        "tqdm>=4.66.0",
        "click>=8.1.0",
        "rich>=13.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
            "jupyter>=1.0.0",
        ],
        "paid_apis": [
            "openai>=1.0.0",
            "google-generativeai>=0.8.0",
        ],
        "full": [
            "selenium>=4.15.0",
            "spacy>=3.7.0",
            "nltk>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fact-check=src.orchestrator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)

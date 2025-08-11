"""
Setup script for Vertex Market Regime System

Cloud-native, GPU-accelerated, adaptive learning framework
for sophisticated options trading regime analysis.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name="vertex-market-regime",
    version="2.0.0",
    author="Market Regime Team",
    author_email="maruthakumar.s@gmail.com",
    description="Cloud-native 8-component adaptive learning system for market regime classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vertex-market-regime",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
        "configs": ["excel/*.xlsx", "yaml/*.yaml", "templates/*"],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies for different use cases
    extras_require={
        "gpu": [
            "cupy-cuda12x>=12.3.0",
            "nvidia-ml-py3>=7.352.0",
        ],
        "dev": [
            "black>=23.11.0",
            "isort>=5.12.0", 
            "mypy>=1.7.0",
            "flake8>=6.1.0",
            "pre-commit>=3.5.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.1.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "financial": [
            "yfinance>=0.2.24",
            "alpha-vantage>=2.3.1",
            "quandl>=3.7.0",
        ]
    },
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "vertex-mr=src.api.fastapi_app:main",
            "vertex-mr-config=configs.excel.excel_parser:main",
            "vertex-mr-train=src.ml.model_trainer:main",
            "vertex-mr-benchmark=scripts.benchmark_performance:main",
        ],
    },
    
    # Project classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: FastAPI",
    ],
    
    # Keywords for PyPI
    keywords=[
        "market-regime", "options-trading", "machine-learning",
        "quantitative-finance", "vertex-ai", "google-cloud",
        "gpu-acceleration", "adaptive-learning", "feature-engineering"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://docs.vertex-market-regime.com",
        "Source": "https://github.com/your-org/vertex-market-regime",
        "Bug Reports": "https://github.com/your-org/vertex-market-regime/issues",
        "Funding": "https://github.com/sponsors/your-org",
    },
    
    # Minimum package requirements
    zip_safe=False,
    
    # Test configuration
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.12.0",
    ],
)
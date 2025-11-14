"""Setup script for MNIST Digit Classification package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

dev_requirements = []
with open("requirements-dev.txt") as f:
    dev_requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#") and not line.startswith("-r")
    ]

setup(
    name="mnist-digit-classifier",
    version="2.0.0",
    author="MNIST Classifier Team",
    author_email="",
    description="A modular neural network implementation for MNIST digit classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyenthusiasts/Handwritten-Digit-Classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
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
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "mnist-train=train:main",
            "mnist-predict=predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

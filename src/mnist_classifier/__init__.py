"""
MNIST Digit Classifier Package

A modular and extensible implementation of a neural network
for handwritten digit classification using the MNIST dataset.
"""

__version__ = "2.0.0"
__author__ = "MNIST Classifier Team"

from .config import Config
from .data_loader import MNISTDataLoader
from .model import MNISTModel
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .visualization import Visualizer

__all__ = [
    "Config",
    "MNISTDataLoader",
    "MNISTModel",
    "ModelTrainer",
    "ModelEvaluator",
    "Visualizer",
]

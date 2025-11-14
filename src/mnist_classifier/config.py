"""
Configuration management for MNIST classifier.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration class for MNIST classifier.

    Attributes:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of training data to use for validation
        learning_rate: Learning rate for optimizer
        hidden_units: Tuple of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        input_shape: Shape of input images
        num_classes: Number of output classes
        optimizer: Optimizer to use
        loss: Loss function to use
        model_dir: Directory to save models
        results_dir: Directory to save results
        data_dir: Directory for data
        random_seed: Random seed for reproducibility
        verbose: Verbosity level
    """

    # Training hyperparameters
    epochs: int = 10
    batch_size: int = 128
    validation_split: float = 0.2
    learning_rate: float = 0.001

    # Model architecture
    hidden_units: tuple = (128, 64)
    dropout_rate: float = 0.2
    input_shape: tuple = (28, 28)
    num_classes: int = 10

    # Compilation parameters
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"

    # Directories
    model_dir: Path = Path("models")
    results_dir: Path = Path("results")
    data_dir: Path = Path("data")

    # Other settings
    random_seed: Optional[int] = 42
    verbose: int = 2

    def __post_init__(self):
        """Ensure directories exist."""
        self.model_dir = Path(self.model_dir)
        self.results_dir = Path(self.results_dir)
        self.data_dir = Path(self.data_dir)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "learning_rate": self.learning_rate,
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "random_seed": self.random_seed,
        }

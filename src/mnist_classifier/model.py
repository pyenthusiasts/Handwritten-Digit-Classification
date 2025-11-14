"""
Neural network model definition for MNIST classification.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

from .config import Config

logger = logging.getLogger(__name__)


class MNISTModel:
    """Neural network model for MNIST digit classification.

    This class encapsulates the creation, compilation, and management
    of a multilayer perceptron for digit classification.
    """

    def __init__(self, config: Config):
        """Initialize the model.

        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None

    def build(self) -> Sequential:
        """Build the neural network architecture.

        Returns:
            Compiled Keras Sequential model
        """
        logger.info("Building neural network model...")

        model = Sequential(name="MNIST_Classifier")

        # Input layer - Flatten 28x28 images
        model.add(Flatten(input_shape=self.config.input_shape, name="input_layer"))

        # Hidden layers
        for i, units in enumerate(self.config.hidden_units):
            model.add(Dense(
                units,
                activation='relu',
                name=f"hidden_layer_{i+1}"
            ))
            model.add(Dropout(
                self.config.dropout_rate,
                name=f"dropout_{i+1}"
            ))

        # Output layer
        model.add(Dense(
            self.config.num_classes,
            activation='softmax',
            name="output_layer"
        ))

        self.model = model
        logger.info(f"Model architecture created with {len(self.config.hidden_units)} hidden layers")

        return model

    def compile(self, learning_rate: Optional[float] = None) -> None:
        """Compile the model with optimizer and loss function.

        Args:
            learning_rate: Learning rate for optimizer. If None, uses config value.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        lr = learning_rate if learning_rate is not None else self.config.learning_rate

        if self.config.optimizer.lower() == 'adam':
            optimizer = Adam(learning_rate=lr)
        else:
            optimizer = self.config.optimizer

        self.model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=['accuracy']
        )

        logger.info(f"Model compiled with {self.config.optimizer} optimizer (lr={lr})")

    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        self.model.summary()

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save the model to disk.

        Args:
            filepath: Path to save the model. If None, uses default path.

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        if filepath is None:
            filepath = self.config.model_dir / "mnist_model.h5"
        else:
            filepath = Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)

        logger.info(f"Model saved to {filepath}")
        return filepath

    def load(self, filepath: Path) -> None:
        """Load a model from disk.

        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_model(self) -> Sequential:
        """Get the underlying Keras model.

        Returns:
            Keras Sequential model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        return self.model

    def count_parameters(self) -> Tuple[int, int]:
        """Count the number of trainable and non-trainable parameters.

        Returns:
            Tuple of (trainable_params, non_trainable_params)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")

        trainable = sum([w.shape.num_elements() for w in self.model.trainable_weights])
        non_trainable = sum([w.shape.num_elements() for w in self.model.non_trainable_weights])

        return trainable, non_trainable

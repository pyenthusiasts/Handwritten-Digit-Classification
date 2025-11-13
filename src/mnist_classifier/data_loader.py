"""
Data loading and preprocessing for MNIST dataset.
"""

import logging
from typing import Tuple

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

logger = logging.getLogger(__name__)


class MNISTDataLoader:
    """Handles loading and preprocessing of MNIST dataset.

    This class provides methods to load the MNIST dataset,
    normalize the images, and convert labels to one-hot encoding.
    """

    def __init__(self, normalize: bool = True, categorical: bool = True):
        """Initialize the data loader.

        Args:
            normalize: Whether to normalize images to [0, 1] range
            categorical: Whether to convert labels to one-hot encoding
        """
        self.normalize = normalize
        self.categorical = categorical
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Load and preprocess the MNIST dataset.

        Returns:
            Tuple containing (X_train, y_train), (X_test, y_test)
        """
        logger.info("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        logger.info(f"Training set size: {X_train.shape[0]} samples")
        logger.info(f"Test set size: {X_test.shape[0]} samples")
        logger.info(f"Image shape: {X_train.shape[1:]}")

        # Preprocess data
        X_train = self._preprocess_images(X_train)
        X_test = self._preprocess_images(X_test)

        if self.categorical:
            y_train = self._preprocess_labels(y_train)
            y_test = self._preprocess_labels(y_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return (X_train, y_train), (X_test, y_test)

    def _preprocess_images(self, images: np.ndarray) -> np.ndarray:
        """Preprocess images by normalizing pixel values.

        Args:
            images: Input images array

        Returns:
            Preprocessed images
        """
        images = images.astype('float32')

        if self.normalize:
            images = images / 255.0
            logger.debug("Images normalized to [0, 1] range")

        return images

    def _preprocess_labels(self, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """Convert labels to one-hot encoding.

        Args:
            labels: Input labels
            num_classes: Number of classes

        Returns:
            One-hot encoded labels
        """
        if self.categorical:
            labels = to_categorical(labels, num_classes)
            logger.debug(f"Labels converted to one-hot encoding with {num_classes} classes")

        return labels

    def get_data_shapes(self) -> dict:
        """Get information about loaded data shapes.

        Returns:
            Dictionary containing shape information
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return {
            "train_samples": self.X_train.shape[0],
            "test_samples": self.X_test.shape[0],
            "image_shape": self.X_train.shape[1:],
            "label_shape": self.y_train.shape[1:] if len(self.y_train.shape) > 1 else (1,),
        }

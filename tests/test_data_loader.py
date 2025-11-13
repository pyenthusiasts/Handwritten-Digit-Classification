"""Tests for data loading functionality."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.data_loader import MNISTDataLoader


class TestMNISTDataLoader:
    """Test cases for MNISTDataLoader class."""

    def test_initialization(self):
        """Test data loader initialization."""
        loader = MNISTDataLoader(normalize=True, categorical=True)
        assert loader.normalize is True
        assert loader.categorical is True
        assert loader.X_train is None
        assert loader.y_train is None

    def test_load_data(self):
        """Test data loading."""
        loader = MNISTDataLoader(normalize=True, categorical=True)
        (X_train, y_train), (X_test, y_test) = loader.load_data()

        # Check shapes
        assert X_train.shape[0] == 60000
        assert X_test.shape[0] == 10000
        assert X_train.shape[1:] == (28, 28)
        assert y_train.shape == (60000, 10)
        assert y_test.shape == (10000, 10)

    def test_normalization(self):
        """Test image normalization."""
        loader = MNISTDataLoader(normalize=True, categorical=True)
        (X_train, _), (X_test, _) = loader.load_data()

        # Check normalization
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
        assert X_test.min() >= 0.0
        assert X_test.max() <= 1.0

    def test_categorical_encoding(self):
        """Test one-hot encoding."""
        loader = MNISTDataLoader(normalize=True, categorical=True)
        (_, y_train), (_, y_test) = loader.load_data()

        # Check one-hot encoding
        assert y_train.shape[1] == 10
        assert y_test.shape[1] == 10
        assert np.all(np.sum(y_train, axis=1) == 1)
        assert np.all(np.sum(y_test, axis=1) == 1)

    def test_get_data_shapes(self):
        """Test data shape information."""
        loader = MNISTDataLoader(normalize=True, categorical=True)
        loader.load_data()

        shapes = loader.get_data_shapes()
        assert shapes['train_samples'] == 60000
        assert shapes['test_samples'] == 10000
        assert shapes['image_shape'] == (28, 28)

    def test_get_data_shapes_before_loading(self):
        """Test error when getting shapes before loading data."""
        loader = MNISTDataLoader()
        with pytest.raises(ValueError):
            loader.get_data_shapes()

"""Tests for training functionality."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.config import Config
from mnist_classifier.model import MNISTModel
from mnist_classifier.trainer import ModelTrainer


class TestModelTrainer:
    """Test cases for ModelTrainer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            epochs=2,
            batch_size=128,
            hidden_units=(64, 32),
            verbose=0,
        )

    @pytest.fixture
    def model(self, config):
        """Create and compile test model."""
        model = MNISTModel(config)
        model.build()
        model.compile()
        return model

    @pytest.fixture
    def trainer(self, model, config):
        """Create test trainer."""
        return ModelTrainer(model, config)

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X_train = np.random.rand(1000, 28, 28).astype('float32')
        y_train = np.eye(10)[np.random.randint(0, 10, 1000)]
        return X_train, y_train

    def test_initialization(self, model, config):
        """Test trainer initialization."""
        trainer = ModelTrainer(model, config)
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.history is None

    def test_train(self, trainer, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        history = trainer.train(X_train, y_train, use_callbacks=False)

        assert history is not None
        assert 'loss' in history
        assert 'accuracy' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['loss']) == trainer.config.epochs

    def test_get_history(self, trainer, sample_data):
        """Test getting training history."""
        X_train, y_train = sample_data

        # Before training
        assert trainer.get_history() is None

        # After training
        trainer.train(X_train, y_train, use_callbacks=False)
        history = trainer.get_history()
        assert history is not None

    def test_get_final_metrics(self, trainer, sample_data):
        """Test getting final metrics."""
        X_train, y_train = sample_data

        # Before training
        assert trainer.get_final_metrics() is None

        # After training
        trainer.train(X_train, y_train, use_callbacks=False)
        metrics = trainer.get_final_metrics()

        assert metrics is not None
        assert 'final_train_loss' in metrics
        assert 'final_train_accuracy' in metrics
        assert 'final_val_loss' in metrics
        assert 'final_val_accuracy' in metrics
        assert 'best_val_accuracy' in metrics
        assert 'best_epoch' in metrics

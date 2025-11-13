"""Tests for model functionality."""

import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.config import Config
from mnist_classifier.model import MNISTModel


class TestMNISTModel:
    """Test cases for MNISTModel class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            epochs=2,
            batch_size=32,
            hidden_units=(64, 32),
            dropout_rate=0.2,
        )

    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return MNISTModel(config)

    def test_initialization(self, config):
        """Test model initialization."""
        model = MNISTModel(config)
        assert model.config == config
        assert model.model is None

    def test_build(self, model):
        """Test model building."""
        keras_model = model.build()
        assert keras_model is not None
        assert model.model is not None
        assert len(model.model.layers) > 0

    def test_compile(self, model):
        """Test model compilation."""
        model.build()
        model.compile()
        # Check optimizer is configured
        assert model.model.optimizer is not None

    def test_compile_without_build(self, model):
        """Test error when compiling without building."""
        with pytest.raises(ValueError):
            model.compile()

    def test_summary(self, model):
        """Test model summary."""
        model.build()
        # Should not raise error
        model.summary()

    def test_summary_without_build(self, model):
        """Test error when getting summary without building."""
        with pytest.raises(ValueError):
            model.summary()

    def test_count_parameters(self, model):
        """Test parameter counting."""
        model.build()
        trainable, non_trainable = model.count_parameters()
        assert trainable > 0
        assert non_trainable >= 0

    def test_save_and_load(self, model):
        """Test model saving and loading."""
        model.build()
        model.compile()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            save_path = Path(tmpdir) / "test_model.h5"
            model.save(save_path)
            assert save_path.exists()

            # Load model
            new_model = MNISTModel(model.config)
            new_model.load(save_path)
            assert new_model.model is not None

    def test_load_nonexistent_file(self, model):
        """Test error when loading nonexistent model."""
        with pytest.raises(FileNotFoundError):
            model.load(Path("/nonexistent/model.h5"))

    def test_get_model(self, model):
        """Test getting underlying Keras model."""
        model.build()
        keras_model = model.get_model()
        assert keras_model is not None

    def test_get_model_without_build(self, model):
        """Test error when getting model without building."""
        with pytest.raises(ValueError):
            model.get_model()

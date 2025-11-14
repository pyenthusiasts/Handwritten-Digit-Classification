"""Tests for configuration functionality."""

import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_default_initialization(self):
        """Test configuration with default values."""
        config = Config()
        assert config.epochs == 10
        assert config.batch_size == 128
        assert config.validation_split == 0.2
        assert config.learning_rate == 0.001
        assert config.hidden_units == (128, 64)
        assert config.dropout_rate == 0.2
        assert config.num_classes == 10

    def test_custom_initialization(self):
        """Test configuration with custom values."""
        config = Config(
            epochs=20,
            batch_size=64,
            learning_rate=0.01,
            hidden_units=(256, 128, 64)
        )
        assert config.epochs == 20
        assert config.batch_size == 64
        assert config.learning_rate == 0.01
        assert config.hidden_units == (256, 128, 64)

    def test_directories_created(self):
        """Test that directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config = Config(
                model_dir=tmpdir / "models",
                results_dir=tmpdir / "results",
                data_dir=tmpdir / "data"
            )
            assert config.model_dir.exists()
            assert config.results_dir.exists()
            assert (config.results_dir / "plots").exists()
            assert (config.results_dir / "logs").exists()
            assert config.data_dir.exists()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config(epochs=15, batch_size=256)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['epochs'] == 15
        assert config_dict['batch_size'] == 256
        assert 'hidden_units' in config_dict
        assert 'dropout_rate' in config_dict

    def test_path_conversion(self):
        """Test that paths are converted to Path objects."""
        config = Config(
            model_dir="models",
            results_dir="results",
            data_dir="data"
        )
        assert isinstance(config.model_dir, Path)
        assert isinstance(config.results_dir, Path)
        assert isinstance(config.data_dir, Path)

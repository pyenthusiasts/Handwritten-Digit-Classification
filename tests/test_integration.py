"""Integration tests for MNIST classifier.

These tests verify the complete workflow from data loading
through training, evaluation, and prediction.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier import (
    Config,
    MNISTDataLoader,
    MNISTModel,
    ModelTrainer,
    ModelEvaluator,
    Visualizer,
)


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return Config(
            epochs=2,
            batch_size=128,
            hidden_units=(64, 32),
            model_dir=temp_dir / "models",
            results_dir=temp_dir / "results",
            verbose=0,
        )

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for testing."""
        X_train = np.random.rand(1000, 28, 28).astype('float32')
        y_train = np.eye(10)[np.random.randint(0, 10, 1000)]
        X_test = np.random.rand(200, 28, 28).astype('float32')
        y_test = np.eye(10)[np.random.randint(0, 10, 200)]
        return (X_train, y_train), (X_test, y_test)

    def test_complete_training_pipeline(self, config, small_dataset):
        """Test complete training pipeline."""
        (X_train, y_train), (X_test, y_test) = small_dataset

        # Build model
        model = MNISTModel(config)
        model.build()
        model.compile()

        # Train
        trainer = ModelTrainer(model, config)
        history = trainer.train(X_train, y_train, use_callbacks=False)

        # Verify training completed
        assert history is not None
        assert len(history['loss']) == config.epochs

        # Evaluate
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        # Verify evaluation
        assert 'test_accuracy' in metrics
        assert 'test_loss' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1

    def test_model_save_and_load(self, config, small_dataset):
        """Test model persistence."""
        (X_train, y_train), (X_test, y_test) = small_dataset

        # Train model
        model = MNISTModel(config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, config)
        trainer.train(X_train, y_train, use_callbacks=False)

        # Make predictions with original model
        evaluator = ModelEvaluator(model)
        original_preds = evaluator.predict(X_test[:10])

        # Save model
        save_path = config.model_dir / "test_model.h5"
        model.save(save_path)

        # Load model
        new_model = MNISTModel(config)
        new_model.load(save_path)

        # Make predictions with loaded model
        new_evaluator = ModelEvaluator(new_model)
        loaded_preds = new_evaluator.predict(X_test[:10])

        # Verify predictions match
        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)

    def test_visualization_pipeline(self, config, small_dataset):
        """Test visualization generation."""
        (X_train, y_train), (X_test, y_test) = small_dataset

        # Train model
        model = MNISTModel(config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, config)
        history = trainer.train(X_train, y_train, use_callbacks=False)

        # Evaluate
        evaluator = ModelEvaluator(model)
        evaluator.evaluate(X_test, y_test)
        cm = evaluator.get_confusion_matrix(X_test, y_test)
        per_class_acc = evaluator.get_per_class_accuracy(X_test, y_test)

        # Create visualizations
        visualizer = Visualizer(config)

        # Training history
        visualizer.plot_training_history(history, save=True)
        assert (config.results_dir / "plots" / "training_history.png").exists()

        # Confusion matrix
        visualizer.plot_confusion_matrix(cm, save=True)
        assert (config.results_dir / "plots" / "confusion_matrix.png").exists()

        # Per-class accuracy
        visualizer.plot_per_class_accuracy(per_class_acc, save=True)
        assert (config.results_dir / "plots" / "per_class_accuracy.png").exists()

    def test_full_mnist_workflow(self, config):
        """Test with actual MNIST dataset (small subset)."""
        # Load real MNIST data
        data_loader = MNISTDataLoader(normalize=True, categorical=True)
        (X_train, y_train), (X_test, y_test) = data_loader.load_data()

        # Use small subset
        X_train = X_train[:1000]
        y_train = y_train[:1000]
        X_test = X_test[:200]
        y_test = y_test[:200]

        # Build and train
        model = MNISTModel(config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, config)
        trainer.train(X_train, y_train, use_callbacks=False)

        # Evaluate
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        # Verify reasonable performance
        assert metrics['test_accuracy'] > 0.5  # Should get better than random

    def test_batch_prediction(self, config, small_dataset):
        """Test batch prediction workflow."""
        (X_train, y_train), (X_test, y_test) = small_dataset

        # Train model
        model = MNISTModel(config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, config)
        trainer.train(X_train, y_train, use_callbacks=False)

        # Batch predictions
        evaluator = ModelEvaluator(model)
        predictions = evaluator.predict(X_test)

        # Verify predictions
        assert predictions.shape == (len(X_test), 10)
        assert np.all((predictions >= 0) & (predictions <= 1))
        assert np.allclose(np.sum(predictions, axis=1), 1.0, atol=1e-5)

    def test_metrics_computation(self, config, small_dataset):
        """Test all metrics computation."""
        (X_train, y_train), (X_test, y_test) = small_dataset

        # Train model
        model = MNISTModel(config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, config)
        trainer.train(X_train, y_train, use_callbacks=False)

        # Compute all metrics
        evaluator = ModelEvaluator(model)

        # Basic metrics
        metrics = evaluator.evaluate(X_test, y_test)
        assert 'test_accuracy' in metrics
        assert 'test_loss' in metrics

        # Confusion matrix
        cm = evaluator.get_confusion_matrix(X_test, y_test)
        assert cm.shape == (10, 10)

        # Top-k accuracy
        top_3_acc = evaluator.get_top_k_accuracy(X_test, y_test, k=3)
        assert 0 <= top_3_acc <= 1

        # Per-class accuracy
        per_class_acc = evaluator.get_per_class_accuracy(X_test, y_test)
        assert len(per_class_acc) == 10

        # Classification report
        report = evaluator.get_classification_report(X_test, y_test)
        assert isinstance(report, str)
        assert len(report) > 0

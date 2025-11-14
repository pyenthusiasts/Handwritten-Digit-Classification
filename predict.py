#!/usr/bin/env python3
"""
Prediction script for MNIST digit classification.

This script loads a trained model and makes predictions on test data.
"""

import sys
import logging
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mnist_classifier import (
    Config,
    MNISTDataLoader,
    MNISTModel,
    ModelEvaluator,
    Visualizer,
)
from mnist_classifier.cli import parse_predict_args, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_predict_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("MNIST DIGIT CLASSIFICATION - PREDICTION")
    logger.info("=" * 60)

    # Check if model exists
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    # Load data
    logger.info("Loading test data...")
    data_loader = MNISTDataLoader(normalize=True, categorical=True)
    (_, _), (X_test, y_test) = data_loader.load_data()

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    config = Config(results_dir=args.output_dir)
    model = MNISTModel(config)
    model.load(args.model_path)

    logger.info("Model loaded successfully!")

    # Create evaluator
    evaluator = ModelEvaluator(model)

    # Evaluate on test set
    logger.info("\nEvaluating model on test set...")
    test_metrics = evaluator.evaluate(X_test, y_test)

    logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"Test Loss: {test_metrics['test_loss']:.4f}")

    # Get samples to predict
    if args.show_misclassified:
        logger.info("\nFinding misclassified samples...")
        images, true_labels, pred_labels, indices = evaluator.get_misclassified_samples(
            X_test, y_test, max_samples=args.num_samples
        )
        logger.info(f"Found {len(indices)} misclassified samples")
    else:
        logger.info(f"\nPredicting {args.num_samples} random samples...")
        indices = np.random.choice(len(X_test), args.num_samples, replace=False)
        images = X_test[indices]
        true_labels = np.argmax(y_test[indices], axis=1)
        pred_labels = evaluator.predict_classes(images)

    # Display predictions
    logger.info("\nPredictions:")
    logger.info("-" * 60)
    for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
        status = "✓" if true == pred else "✗"
        logger.info(f"Sample {i+1}: True={true}, Predicted={pred} {status}")

    # Save visualizations if requested
    if args.save_predictions:
        logger.info("\nSaving prediction visualizations...")
        config.results_dir = args.output_dir
        visualizer = Visualizer(config)

        filename = "misclassified_predictions.png" if args.show_misclassified else "predictions.png"
        visualizer.plot_sample_predictions(
            images,
            true_labels,
            pred_labels,
            num_samples=len(images),
            save=True,
            filename=filename
        )
        logger.info(f"Visualizations saved to {config.results_dir / 'plots'}")

    # Compute accuracy for these samples
    accuracy = np.mean(true_labels == pred_labels)
    logger.info("\n" + "=" * 60)
    logger.info(f"Accuracy on selected samples: {accuracy:.4f} ({np.sum(true_labels == pred_labels)}/{len(true_labels)})")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nPrediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError during prediction: {e}", exc_info=True)
        sys.exit(1)

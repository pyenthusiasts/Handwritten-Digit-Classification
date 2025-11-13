#!/usr/bin/env python3
"""
Main training script for MNIST digit classification.

This script provides a command-line interface for training
the MNIST digit classifier with configurable parameters.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mnist_classifier import (
    Config,
    MNISTDataLoader,
    MNISTModel,
    ModelTrainer,
    ModelEvaluator,
    Visualizer,
)
from mnist_classifier.cli import parse_train_args, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_train_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("MNIST DIGIT CLASSIFICATION - TRAINING")
    logger.info("=" * 60)

    # Create configuration
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        hidden_units=tuple(args.hidden_units),
        dropout_rate=args.dropout_rate,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        random_seed=args.seed,
        verbose=2 if args.verbose else 1,
    )

    logger.info("Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")

    # Load data
    logger.info("\nLoading data...")
    data_loader = MNISTDataLoader(normalize=True, categorical=True)
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()

    data_info = data_loader.get_data_shapes()
    logger.info(f"Data shapes: {data_info}")

    # Build model
    logger.info("\nBuilding model...")
    model = MNISTModel(config)
    model.build()
    model.compile()

    # Print model summary
    if args.verbose:
        model.summary()

    trainable, non_trainable = model.count_parameters()
    logger.info(f"Trainable parameters: {trainable:,}")
    logger.info(f"Non-trainable parameters: {non_trainable:,}")

    # Train model
    logger.info("\nTraining model...")
    trainer = ModelTrainer(model, config)
    history = trainer.train(
        X_train,
        y_train,
        use_callbacks=not args.no_callbacks,
        early_stopping=args.early_stopping,
    )

    # Get final metrics
    final_metrics = trainer.get_final_metrics()
    logger.info("\nTraining completed!")
    logger.info("Final Metrics:")
    for key, value in final_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    evaluator = ModelEvaluator(model)
    test_metrics = evaluator.evaluate(X_test, y_test)

    # Additional metrics
    logger.info("\nComputing additional metrics...")
    top_3_acc = evaluator.get_top_k_accuracy(X_test, y_test, k=3)
    per_class_acc = evaluator.get_per_class_accuracy(X_test, y_test)

    # Classification report
    logger.info("\nClassification Report:")
    report = evaluator.get_classification_report(X_test, y_test)
    print(report)

    # Visualizations
    if not args.no_plots:
        logger.info("\nGenerating visualizations...")
        visualizer = Visualizer(config)

        # Training history
        visualizer.plot_training_history(history)

        # Confusion matrix
        cm = evaluator.get_confusion_matrix(X_test, y_test)
        visualizer.plot_confusion_matrix(cm)

        # Per-class accuracy
        visualizer.plot_per_class_accuracy(per_class_acc)

        # Sample predictions
        predictions = evaluator.predict_classes(X_test[:20])
        import numpy as np
        true_labels = np.argmax(y_test[:20], axis=1)
        visualizer.plot_sample_predictions(
            X_test[:20],
            true_labels,
            predictions,
            num_samples=20,
            filename="sample_predictions.png"
        )

        # Misclassified samples
        mis_images, mis_true, mis_pred, _ = evaluator.get_misclassified_samples(
            X_test, y_test, max_samples=20
        )
        if len(mis_images) > 0:
            visualizer.plot_misclassified_samples(
                mis_images,
                mis_true,
                mis_pred,
                num_samples=min(20, len(mis_images))
            )

        # Summary report
        summary_metrics = {
            'config': config.to_dict(),
            **final_metrics,
            **test_metrics,
            'top_3_accuracy': top_3_acc,
        }
        report_text = visualizer.create_summary_report(summary_metrics)
        print("\n" + report_text)

    # Save final model
    logger.info("\nSaving final model...")
    model_path = model.save(config.model_dir / "final_model.h5")
    logger.info(f"Model saved to {model_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    logger.info(f"Results saved to: {config.results_dir}")
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError during training: {e}", exc_info=True)
        sys.exit(1)

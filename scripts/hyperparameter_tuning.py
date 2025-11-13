#!/usr/bin/env python3
"""
Hyperparameter tuning for MNIST classifier.

Performs grid search or random search over hyperparameters
to find optimal model configuration.
"""

import argparse
import sys
import logging
from pathlib import Path
import json
from datetime import datetime
from itertools import product
import random

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier import (
    Config,
    MNISTDataLoader,
    MNISTModel,
    ModelTrainer,
    ModelEvaluator,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning utility."""

    def __init__(self, param_grid: dict, search_type: str = 'grid'):
        """Initialize tuner.

        Args:
            param_grid: Dictionary of parameters to search
            search_type: 'grid' or 'random'
        """
        self.param_grid = param_grid
        self.search_type = search_type
        self.results = []
        self.best_params = None
        self.best_score = 0.0

    def get_param_combinations(self, max_trials: int = None):
        """Generate parameter combinations.

        Args:
            max_trials: Maximum number of trials (for random search)

        Returns:
            List of parameter dictionaries
        """
        if self.search_type == 'grid':
            # Generate all combinations
            keys = self.param_grid.keys()
            values = self.param_grid.values()
            combinations = [dict(zip(keys, v)) for v in product(*values)]

            logger.info(f"Grid search: {len(combinations)} combinations")
            return combinations

        elif self.search_type == 'random':
            # Generate random combinations
            if max_trials is None:
                max_trials = 20

            combinations = []
            for _ in range(max_trials):
                combo = {k: random.choice(v) for k, v in self.param_grid.items()}
                combinations.append(combo)

            logger.info(f"Random search: {max_trials} trials")
            return combinations

    def train_and_evaluate(self, params: dict, X_train, y_train, X_test, y_test):
        """Train model with given parameters and evaluate.

        Args:
            params: Parameter dictionary
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels

        Returns:
            Dictionary with results
        """
        # Create config with parameters
        config = Config(
            epochs=params.get('epochs', 10),
            batch_size=params.get('batch_size', 128),
            learning_rate=params.get('learning_rate', 0.001),
            hidden_units=tuple(params.get('hidden_units', [128, 64])),
            dropout_rate=params.get('dropout_rate', 0.2),
            verbose=0,
        )

        # Build and train model
        model = MNISTModel(config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, config)
        history = trainer.train(X_train, y_train, use_callbacks=False)

        # Evaluate
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(X_test, y_test)

        # Get final training metrics
        final_metrics = trainer.get_final_metrics()

        return {
            'params': params,
            'test_accuracy': metrics['test_accuracy'],
            'test_loss': metrics['test_loss'],
            'val_accuracy': final_metrics['final_val_accuracy'],
            'val_loss': final_metrics['final_val_loss'],
            'best_val_accuracy': final_metrics['best_val_accuracy'],
            'best_epoch': final_metrics['best_epoch'],
        }

    def tune(self, X_train, y_train, X_test, y_test, max_trials: int = None):
        """Run hyperparameter tuning.

        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            max_trials: Maximum number of trials (for random search)
        """
        combinations = self.get_param_combinations(max_trials)
        total = len(combinations)

        logger.info("=" * 60)
        logger.info(f"Starting hyperparameter tuning ({self.search_type} search)")
        logger.info("=" * 60)

        for i, params in enumerate(combinations, 1):
            logger.info(f"\nTrial {i}/{total}")
            logger.info(f"Parameters: {params}")

            try:
                result = self.train_and_evaluate(params, X_train, y_train, X_test, y_test)
                self.results.append(result)

                score = result['test_accuracy']
                logger.info(f"Test Accuracy: {score:.4f}")

                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    logger.info("★ New best model!")

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                continue

        logger.info("\n" + "=" * 60)
        logger.info("TUNING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nBest accuracy: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

    def save_results(self, output_path: Path):
        """Save tuning results to JSON.

        Args:
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            'search_type': self.search_type,
            'param_grid': self.param_grid,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    def print_top_k(self, k: int = 5):
        """Print top k results.

        Args:
            k: Number of top results to print
        """
        if not self.results:
            return

        sorted_results = sorted(self.results, key=lambda x: x['test_accuracy'], reverse=True)

        logger.info(f"\nTop {k} configurations:")
        logger.info("-" * 80)

        for i, result in enumerate(sorted_results[:k], 1):
            logger.info(f"\n#{i} - Test Accuracy: {result['test_accuracy']:.4f}")
            logger.info(f"Parameters: {result['params']}")
            logger.info(f"Val Accuracy: {result['val_accuracy']:.4f}, "
                       f"Best Val Accuracy: {result['best_val_accuracy']:.4f}")


def main():
    """Main tuning function."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for MNIST classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--search-type',
        choices=['grid', 'random'],
        default='random',
        help='Search type'
    )
    parser.add_argument(
        '--max-trials',
        type=int,
        default=10,
        help='Maximum number of trials (for random search)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/tuning_results.json'),
        help='Output file for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick tuning with fewer epochs'
    )

    args = parser.parse_args()

    # Define parameter grid
    if args.quick:
        param_grid = {
            'epochs': [3, 5],
            'batch_size': [128, 256],
            'learning_rate': [0.001, 0.01],
            'hidden_units': [[128, 64], [256, 128]],
            'dropout_rate': [0.2, 0.3],
        }
    else:
        param_grid = {
            'epochs': [10, 15, 20],
            'batch_size': [64, 128, 256],
            'learning_rate': [0.0001, 0.001, 0.01],
            'hidden_units': [
                [64, 32],
                [128, 64],
                [256, 128, 64],
                [512, 256, 128],
            ],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        }

    # Load data
    logger.info("Loading MNIST dataset...")
    data_loader = MNISTDataLoader(normalize=True, categorical=True)
    (X_train, y_train), (X_test, y_test) = data_loader.load_data()

    # Use smaller subset for faster tuning
    logger.info("Using subset of data for faster tuning...")
    X_train = X_train[:10000]
    y_train = y_train[:10000]

    # Create tuner and run
    tuner = HyperparameterTuner(param_grid, search_type=args.search_type)
    tuner.tune(X_train, y_train, X_test, y_test, max_trials=args.max_trials)

    # Print top results
    tuner.print_top_k(k=5)

    # Save results
    tuner.save_results(args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nTuning interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTuning failed: {e}", exc_info=True)
        sys.exit(1)

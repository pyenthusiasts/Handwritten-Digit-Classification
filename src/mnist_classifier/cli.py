"""
Command-line interface for MNIST classifier.
"""

import argparse
import logging
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_train_parser() -> argparse.ArgumentParser:
    """Create argument parser for training script.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Train MNIST digit classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of training data for validation'
    )

    # Model architecture
    parser.add_argument(
        '--hidden-units',
        type=int,
        nargs='+',
        default=[128, 64],
        help='Hidden layer sizes'
    )
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.2,
        help='Dropout rate for regularization'
    )

    # Callbacks and features
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--no-callbacks',
        action='store_true',
        help='Disable training callbacks'
    )

    # Directories
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=Path('models'),
        help='Directory to save models'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results'),
        help='Directory to save results'
    )

    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )

    return parser


def create_predict_parser() -> argparse.ArgumentParser:
    """Create argument parser for prediction script.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Make predictions using trained MNIST model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of test samples to predict'
    )
    parser.add_argument(
        '--show-misclassified',
        action='store_true',
        help='Show only misclassified samples'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction visualizations'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/predictions'),
        help='Directory to save prediction outputs'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser


def parse_train_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed arguments
    """
    parser = create_train_parser()
    return parser.parse_args()


def parse_predict_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction.

    Returns:
        Parsed arguments
    """
    parser = create_predict_parser()
    return parser.parse_args()

"""
Visualization utilities for MNIST classifier.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from .config import Config

logger = logging.getLogger(__name__)


class Visualizer:
    """Handles visualization of training results and predictions.

    This class provides methods to create and save various plots
    for model analysis and result presentation.
    """

    def __init__(self, config: Config):
        """Initialize the visualizer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.plot_dir = config.results_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_training_history(
        self,
        history: Dict[str, Any],
        save: bool = True,
        filename: str = "training_history.png"
    ) -> Optional[Figure]:
        """Plot training and validation accuracy/loss.

        Args:
            history: Training history dictionary
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib Figure object if not saving, None otherwise
        """
        logger.info("Plotting training history...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # Plot loss
        axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = self.plot_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
            plt.close()
            return None
        else:
            return fig

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save: bool = True,
        filename: str = "confusion_matrix.png"
    ) -> Optional[Figure]:
        """Plot confusion matrix as heatmap.

        Args:
            confusion_matrix: Confusion matrix array
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib Figure object if not saving, None otherwise
        """
        logger.info("Plotting confusion matrix...")

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        plt.tight_layout()

        if save:
            save_path = self.plot_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
            plt.close()
            return None
        else:
            return fig

    def plot_sample_predictions(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        num_samples: int = 10,
        save: bool = True,
        filename: str = "sample_predictions.png"
    ) -> Optional[Figure]:
        """Plot sample predictions with images.

        Args:
            images: Array of images
            true_labels: True labels
            predicted_labels: Predicted labels
            num_samples: Number of samples to plot
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib Figure object if not saving, None otherwise
        """
        logger.info(f"Plotting {num_samples} sample predictions...")

        num_samples = min(num_samples, len(images))
        cols = 5
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten() if num_samples > 1 else [axes]

        for i in range(num_samples):
            ax = axes[i]
            ax.imshow(images[i], cmap='gray')

            # Color code: green for correct, red for incorrect
            color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
            ax.set_title(
                f'True: {true_labels[i]}\nPred: {predicted_labels[i]}',
                color=color,
                fontweight='bold'
            )
            ax.axis('off')

        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save:
            save_path = self.plot_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sample predictions plot saved to {save_path}")
            plt.close()
            return None
        else:
            return fig

    def plot_per_class_accuracy(
        self,
        per_class_acc: Dict[int, float],
        save: bool = True,
        filename: str = "per_class_accuracy.png"
    ) -> Optional[Figure]:
        """Plot per-class accuracy as bar chart.

        Args:
            per_class_acc: Dictionary mapping class to accuracy
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib Figure object if not saving, None otherwise
        """
        logger.info("Plotting per-class accuracy...")

        fig, ax = plt.subplots(figsize=(10, 6))

        classes = sorted(per_class_acc.keys())
        accuracies = [per_class_acc[c] for c in classes]

        bars = ax.bar(classes, accuracies, color='steelblue', alpha=0.8)

        # Color bars based on accuracy
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc < 0.9:
                bar.set_color('orangered')
            elif acc < 0.95:
                bar.set_color('orange')

        ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.set_xticks(classes)
        ax.grid(True, alpha=0.3, axis='y')

        # Add accuracy values on top of bars
        for i, (cls, acc) in enumerate(zip(classes, accuracies)):
            ax.text(cls, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.plot_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class accuracy plot saved to {save_path}")
            plt.close()
            return None
        else:
            return fig

    def plot_misclassified_samples(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        num_samples: int = 10,
        save: bool = True,
        filename: str = "misclassified_samples.png"
    ) -> Optional[Figure]:
        """Plot misclassified samples.

        Args:
            images: Array of misclassified images
            true_labels: True labels
            predicted_labels: Predicted labels
            num_samples: Number of samples to plot
            save: Whether to save the plot
            filename: Filename for saved plot

        Returns:
            Matplotlib Figure object if not saving, None otherwise
        """
        logger.info(f"Plotting {num_samples} misclassified samples...")

        return self.plot_sample_predictions(
            images,
            true_labels,
            predicted_labels,
            num_samples,
            save,
            filename
        )

    def create_summary_report(
        self,
        metrics: Dict[str, Any],
        save: bool = True,
        filename: str = "summary_report.txt"
    ) -> str:
        """Create a text summary report.

        Args:
            metrics: Dictionary of metrics to include
            save: Whether to save the report
            filename: Filename for saved report

        Returns:
            Summary report string
        """
        logger.info("Creating summary report...")

        report_lines = [
            "=" * 60,
            "MNIST DIGIT CLASSIFICATION - SUMMARY REPORT",
            "=" * 60,
            "",
            "MODEL CONFIGURATION:",
            "-" * 60,
        ]

        if 'config' in metrics:
            config = metrics['config']
            for key, value in config.items():
                report_lines.append(f"  {key}: {value}")

        report_lines.extend([
            "",
            "PERFORMANCE METRICS:",
            "-" * 60,
        ])

        for key, value in metrics.items():
            if key != 'config':
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")

        report_lines.extend([
            "",
            "=" * 60,
        ])

        report = "\n".join(report_lines)

        if save:
            save_path = self.config.results_dir / filename
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Summary report saved to {save_path}")

        return report

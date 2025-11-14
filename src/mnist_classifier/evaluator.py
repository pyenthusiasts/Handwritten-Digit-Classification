"""
Model evaluation functionality.
"""

import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .model import MNISTModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles evaluation of trained MNIST model.

    This class provides methods to evaluate model performance,
    generate predictions, and compute various metrics.
    """

    def __init__(self, model: MNISTModel):
        """Initialize the evaluator.

        Args:
            model: Trained MNISTModel instance
        """
        self.model = model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 0
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            verbose: Verbosity level

        Returns:
            Dictionary containing loss and accuracy
        """
        logger.info("Evaluating model on test set...")

        test_loss, test_accuracy = self.model.get_model().evaluate(
            X_test,
            y_test,
            verbose=verbose
        )

        metrics = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
        }

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input data.

        Args:
            X: Input images

        Returns:
            Predicted class probabilities
        """
        return self.model.get_model().predict(X, verbose=0)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input data.

        Args:
            X: Input images

        Returns:
            Predicted class labels
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)

    def get_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)

        Returns:
            Confusion matrix
        """
        y_pred = self.predict_classes(X_test)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        logger.info("Confusion matrix computed")

        return cm

    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """Generate classification report.

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            target_names: List of target class names

        Returns:
            Classification report string
        """
        y_pred = self.predict_classes(X_test)
        y_true = np.argmax(y_test, axis=1)

        if target_names is None:
            target_names = [str(i) for i in range(10)]

        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4
        )

        logger.info("Classification report generated")
        return report

    def get_misclassified_samples(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get misclassified samples.

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            max_samples: Maximum number of samples to return

        Returns:
            Tuple of (images, true_labels, predicted_labels, indices)
        """
        y_pred = self.predict_classes(X_test)
        y_true = np.argmax(y_test, axis=1)

        # Find misclassified indices
        misclassified_idx = np.where(y_pred != y_true)[0]

        # Limit to max_samples
        if len(misclassified_idx) > max_samples:
            misclassified_idx = misclassified_idx[:max_samples]

        logger.info(f"Found {len(misclassified_idx)} misclassified samples (showing up to {max_samples})")

        return (
            X_test[misclassified_idx],
            y_true[misclassified_idx],
            y_pred[misclassified_idx],
            misclassified_idx
        )

    def get_top_k_accuracy(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        k: int = 3
    ) -> float:
        """Compute top-k accuracy.

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy
        """
        predictions = self.predict(X_test)
        y_true = np.argmax(y_test, axis=1)

        # Get top k predictions
        top_k_preds = np.argsort(predictions, axis=1)[:, -k:]

        # Check if true label is in top k
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        top_k_acc = np.mean(correct)

        logger.info(f"Top-{k} accuracy: {top_k_acc:.4f}")
        return float(top_k_acc)

    def get_per_class_accuracy(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[int, float]:
        """Compute per-class accuracy.

        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)

        Returns:
            Dictionary mapping class to accuracy
        """
        y_pred = self.predict_classes(X_test)
        y_true = np.argmax(y_test, axis=1)

        per_class_acc = {}
        for class_idx in range(10):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
                per_class_acc[class_idx] = float(class_accuracy)

        logger.info("Per-class accuracy computed")
        return per_class_acc

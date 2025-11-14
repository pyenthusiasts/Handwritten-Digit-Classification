"""
Model training functionality.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from .config import Config
from .model import MNISTModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of MNIST classification model.

    This class manages the training process, including callbacks,
    history tracking, and saving training metadata.
    """

    def __init__(self, model: MNISTModel, config: Config):
        """Initialize the trainer.

        Args:
            model: MNISTModel instance
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.history = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        use_callbacks: bool = True,
        early_stopping: bool = False,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X_train: Training images
            y_train: Training labels
            use_callbacks: Whether to use training callbacks
            early_stopping: Whether to use early stopping

        Returns:
            Training history dictionary
        """
        logger.info("Starting model training...")
        logger.info(f"Training for {self.config.epochs} epochs with batch size {self.config.batch_size}")

        callbacks = []
        if use_callbacks:
            callbacks = self._create_callbacks(early_stopping)

        # Train the model
        self.history = self.model.get_model().fit(
            X_train,
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )

        logger.info("Training completed!")

        # Save training history
        self._save_history()

        return self.history.history

    def _create_callbacks(self, early_stopping: bool = False) -> list:
        """Create training callbacks.

        Args:
            early_stopping: Whether to include early stopping

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # Model checkpoint - save best model
        checkpoint_path = self.config.model_dir / "best_model.h5"
        callbacks.append(ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ))

        # CSV logger
        log_path = self.config.results_dir / "logs" / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        callbacks.append(CSVLogger(str(log_path)))

        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ))

        # Early stopping (optional)
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ))
            logger.info("Early stopping enabled with patience=5")

        return callbacks

    def _save_history(self) -> None:
        """Save training history to JSON file."""
        if self.history is None:
            logger.warning("No training history to save")
            return

        history_path = self.config.results_dir / "logs" / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert numpy arrays to lists for JSON serialization
        history_dict = {
            key: [float(val) for val in values]
            for key, values in self.history.history.items()
        }

        # Add metadata
        history_dict['metadata'] = {
            'epochs_completed': len(history_dict['loss']),
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

    def get_history(self) -> Optional[Dict[str, Any]]:
        """Get the training history.

        Returns:
            Training history dictionary or None if not trained
        """
        return self.history.history if self.history else None

    def get_final_metrics(self) -> Optional[Dict[str, float]]:
        """Get final training and validation metrics.

        Returns:
            Dictionary containing final metrics
        """
        if self.history is None:
            return None

        history = self.history.history
        return {
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': max(history['val_accuracy']),
            'best_epoch': int(np.argmax(history['val_accuracy'])) + 1,
        }

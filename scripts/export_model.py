#!/usr/bin/env python3
"""
Export trained MNIST model to different formats.

Supports export to:
- ONNX (Open Neural Network Exchange)
- TensorFlow Lite
- SavedModel format
"""

import argparse
import sys
import logging
from pathlib import Path

import tensorflow as tf
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier import Config, MNISTModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_to_onnx(model_path: Path, output_path: Path):
    """Export model to ONNX format.

    Args:
        model_path: Path to the Keras model
        output_path: Path to save ONNX model
    """
    try:
        import tf2onnx
        import onnx
    except ImportError:
        logger.error("tf2onnx not installed. Install with: pip install tf2onnx onnx")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    logger.info("Converting to ONNX format...")
    spec = (tf.TensorSpec((None, 28, 28), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(output_path)
    )

    logger.info(f"Model exported to ONNX: {output_path}")
    logger.info(f"ONNX model size: {output_path.stat().st_size / 1024:.2f} KB")


def export_to_tflite(model_path: Path, output_path: Path, quantize: bool = False):
    """Export model to TensorFlow Lite format.

    Args:
        model_path: Path to the Keras model
        output_path: Path to save TFLite model
        quantize: Whether to apply quantization
    """
    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    logger.info("Converting to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        logger.info("Applying quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    logger.info(f"Model exported to TFLite: {output_path}")
    logger.info(f"TFLite model size: {output_path.stat().st_size / 1024:.2f} KB")

    if quantize:
        original_size = model_path.stat().st_size / 1024
        tflite_size = output_path.stat().st_size / 1024
        reduction = ((original_size - tflite_size) / original_size) * 100
        logger.info(f"Size reduction: {reduction:.1f}%")


def export_to_savedmodel(model_path: Path, output_dir: Path):
    """Export model to TensorFlow SavedModel format.

    Args:
        model_path: Path to the Keras model
        output_dir: Directory to save SavedModel
    """
    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    logger.info(f"Exporting to SavedModel format: {output_dir}")
    tf.saved_model.save(model, str(output_dir))

    logger.info(f"Model exported to SavedModel: {output_dir}")


def verify_onnx_model(onnx_path: Path, keras_model_path: Path, num_samples: int = 10):
    """Verify ONNX model outputs match Keras model.

    Args:
        onnx_path: Path to ONNX model
        keras_model_path: Path to Keras model
        num_samples: Number of samples to test
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed. Skipping verification.")
        return

    logger.info("Verifying ONNX model...")

    # Load models
    keras_model = tf.keras.models.load_model(keras_model_path)
    ort_session = ort.InferenceSession(str(onnx_path))

    # Generate random test data
    test_data = np.random.rand(num_samples, 28, 28).astype(np.float32)

    # Get predictions from Keras model
    keras_preds = keras_model.predict(test_data, verbose=0)

    # Get predictions from ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: test_data}
    ort_preds = ort_session.run(None, ort_inputs)[0]

    # Compare predictions
    max_diff = np.max(np.abs(keras_preds - ort_preds))
    logger.info(f"Maximum difference between predictions: {max_diff:.6f}")

    if max_diff < 1e-5:
        logger.info("✓ ONNX model verification passed!")
    else:
        logger.warning(f"⚠ Large difference detected: {max_diff}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(
        description='Export MNIST model to different formats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('models/best_model.h5'),
        help='Path to the trained Keras model'
    )
    parser.add_argument(
        '--format',
        choices=['onnx', 'tflite', 'savedmodel', 'all'],
        default='all',
        help='Export format'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('models/exported'),
        help='Output directory for exported models'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply quantization (TFLite only)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify exported models'
    )

    args = parser.parse_args()

    # Check if model exists
    if not args.model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MNIST MODEL EXPORT")
    logger.info("=" * 60)

    # Export based on format
    if args.format in ['onnx', 'all']:
        onnx_path = args.output_dir / 'model.onnx'
        export_to_onnx(args.model_path, onnx_path)
        if args.verify:
            verify_onnx_model(onnx_path, args.model_path)

    if args.format in ['tflite', 'all']:
        tflite_path = args.output_dir / 'model.tflite'
        export_to_tflite(args.model_path, tflite_path, quantize=args.quantize)

        if args.quantize:
            tflite_quant_path = args.output_dir / 'model_quantized.tflite'
            logger.info("\nExporting quantized version...")
            export_to_tflite(args.model_path, tflite_quant_path, quantize=True)

    if args.format in ['savedmodel', 'all']:
        savedmodel_dir = args.output_dir / 'savedmodel'
        export_to_savedmodel(args.model_path, savedmodel_dir)

    logger.info("\n" + "=" * 60)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Exported models saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nExport interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nExport failed: {e}", exc_info=True)
        sys.exit(1)

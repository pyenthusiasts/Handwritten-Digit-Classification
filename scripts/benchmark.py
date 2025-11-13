#!/usr/bin/env python3
"""
Benchmark MNIST model performance.

Measures:
- Training time
- Inference time
- Memory usage
- Model size
- Throughput (predictions/second)
"""

import argparse
import sys
import time
import logging
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

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


class Benchmark:
    """Model benchmarking utility."""

    def __init__(self, config: Config):
        """Initialize benchmark.

        Args:
            config: Configuration object
        """
        self.config = config
        self.results = {}

    def benchmark_data_loading(self, runs: int = 3):
        """Benchmark data loading time.

        Args:
            runs: Number of runs to average
        """
        logger.info("Benchmarking data loading...")
        times = []

        for i in range(runs):
            start = time.time()
            data_loader = MNISTDataLoader(normalize=True, categorical=True)
            (X_train, y_train), (X_test, y_test) = data_loader.load_data()
            elapsed = time.time() - start
            times.append(elapsed)
            logger.info(f"Run {i+1}/{runs}: {elapsed:.3f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)

        self.results['data_loading'] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'runs': runs,
        }

        logger.info(f"Average data loading time: {avg_time:.3f}s (±{std_time:.3f}s)")
        return X_train, y_train, X_test, y_test

    def benchmark_model_building(self, runs: int = 10):
        """Benchmark model building time.

        Args:
            runs: Number of runs to average
        """
        logger.info("\nBenchmarking model building...")
        times = []

        for i in range(runs):
            start = time.time()
            model = MNISTModel(self.config)
            model.build()
            model.compile()
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        self.results['model_building'] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'runs': runs,
        }

        logger.info(f"Average model building time: {avg_time:.4f}s (±{std_time:.4f}s)")
        return model

    def benchmark_training(self, X_train, y_train):
        """Benchmark training time.

        Args:
            X_train: Training data
            y_train: Training labels
        """
        logger.info("\nBenchmarking training...")

        model = MNISTModel(self.config)
        model.build()
        model.compile()

        trainer = ModelTrainer(model, self.config)

        start = time.time()
        history = trainer.train(X_train, y_train, use_callbacks=False)
        elapsed = time.time() - start

        time_per_epoch = elapsed / self.config.epochs
        samples_per_second = len(X_train) / elapsed

        self.results['training'] = {
            'total_time': elapsed,
            'time_per_epoch': time_per_epoch,
            'samples_per_second': samples_per_second,
            'epochs': self.config.epochs,
        }

        logger.info(f"Total training time: {elapsed:.2f}s")
        logger.info(f"Time per epoch: {time_per_epoch:.2f}s")
        logger.info(f"Throughput: {samples_per_second:.0f} samples/s")

        return model

    def benchmark_inference(self, model, X_test, batch_sizes=[1, 32, 128, 256]):
        """Benchmark inference time with different batch sizes.

        Args:
            model: Trained model
            X_test: Test data
            batch_sizes: List of batch sizes to test
        """
        logger.info("\nBenchmarking inference...")

        inference_results = {}

        for batch_size in batch_sizes:
            # Prepare batch
            X_batch = X_test[:batch_size]

            # Warmup
            for _ in range(5):
                model.get_model().predict(X_batch, verbose=0)

            # Benchmark
            num_runs = 100
            times = []

            for _ in range(num_runs):
                start = time.time()
                model.get_model().predict(X_batch, verbose=0)
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = batch_size / avg_time

            inference_results[batch_size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'throughput': throughput,
                'latency_ms': avg_time * 1000,
            }

            logger.info(f"Batch size {batch_size:3d}: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms), "
                       f"{throughput:.0f} samples/s")

        self.results['inference'] = inference_results

    def benchmark_memory(self, model):
        """Benchmark memory usage.

        Args:
            model: Model to benchmark
        """
        logger.info("\nBenchmarking memory usage...")

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            self.results['memory'] = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
            }

            logger.info(f"Memory usage (RSS): {memory_info.rss / 1024 / 1024:.2f} MB")
            logger.info(f"Memory usage (VMS): {memory_info.vms / 1024 / 1024:.2f} MB")

        except ImportError:
            logger.warning("psutil not installed, skipping memory benchmark")

    def benchmark_model_size(self, model):
        """Benchmark model size.

        Args:
            model: Model to benchmark
        """
        logger.info("\nBenchmarking model size...")

        # Save model temporarily
        temp_path = self.config.model_dir / "temp_benchmark.h5"
        model.save(temp_path)

        size_mb = temp_path.stat().st_size / 1024 / 1024
        trainable, non_trainable = model.count_parameters()

        self.results['model_size'] = {
            'file_size_mb': size_mb,
            'trainable_params': trainable,
            'non_trainable_params': non_trainable,
            'total_params': trainable + non_trainable,
        }

        logger.info(f"Model file size: {size_mb:.2f} MB")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"Total parameters: {trainable + non_trainable:,}")

        # Clean up
        temp_path.unlink()

    def save_results(self, output_path: Path):
        """Save benchmark results to JSON.

        Args:
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj

        results = convert_types(self.results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    def print_summary(self):
        """Print benchmark summary."""
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)

        if 'data_loading' in self.results:
            logger.info(f"\nData Loading: {self.results['data_loading']['avg_time']:.3f}s")

        if 'model_building' in self.results:
            logger.info(f"Model Building: {self.results['model_building']['avg_time']:.4f}s")

        if 'training' in self.results:
            t = self.results['training']
            logger.info(f"\nTraining ({t['epochs']} epochs):")
            logger.info(f"  Total time: {t['total_time']:.2f}s")
            logger.info(f"  Per epoch: {t['time_per_epoch']:.2f}s")
            logger.info(f"  Throughput: {t['samples_per_second']:.0f} samples/s")

        if 'inference' in self.results:
            logger.info("\nInference:")
            for batch_size, metrics in self.results['inference'].items():
                logger.info(f"  Batch {batch_size}: {metrics['latency_ms']:.2f}ms, "
                           f"{metrics['throughput']:.0f} samples/s")

        if 'model_size' in self.results:
            s = self.results['model_size']
            logger.info(f"\nModel Size:")
            logger.info(f"  File size: {s['file_size_mb']:.2f} MB")
            logger.info(f"  Parameters: {s['total_params']:,}")

        if 'memory' in self.results:
            logger.info(f"\nMemory Usage: {self.results['memory']['rss_mb']:.2f} MB")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description='Benchmark MNIST model performance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--output', type=Path, default=Path('results/benchmark_results.json'),
                       help='Output file for results')
    parser.add_argument('--skip-training', action='store_true', help='Skip training benchmark')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MNIST MODEL BENCHMARK")
    logger.info("=" * 60)

    # Create config
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )

    # Run benchmarks
    benchmark = Benchmark(config)

    # Data loading
    X_train, y_train, X_test, y_test = benchmark.benchmark_data_loading()

    # Model building
    model = benchmark.benchmark_model_building()

    # Training
    if not args.skip_training:
        model = benchmark.training(X_train, y_train)
    else:
        # Train with minimal config for other benchmarks
        logger.info("\nTraining model for inference benchmark (2 epochs)...")
        quick_config = Config(epochs=2, batch_size=128, verbose=0)
        model = MNISTModel(quick_config)
        model.build()
        model.compile()
        trainer = ModelTrainer(model, quick_config)
        trainer.train(X_train, y_train, use_callbacks=False)

    # Inference
    benchmark.benchmark_inference(model, X_test)

    # Model size
    benchmark.benchmark_model_size(model)

    # Memory
    benchmark.benchmark_memory(model)

    # Save and print results
    benchmark.save_results(args.output)
    benchmark.print_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}", exc_info=True)
        sys.exit(1)

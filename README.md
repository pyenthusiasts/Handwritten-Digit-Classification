# Handwritten Digit Classification with Neural Network

[![CI/CD Pipeline](https://github.com/pyenthusiasts/Handwritten-Digit-Classification/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/pyenthusiasts/Handwritten-Digit-Classification/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional, modular, and extensible implementation of a neural network for classifying handwritten digits from the MNIST dataset. This project demonstrates best practices in machine learning engineering, including proper code organization, testing, logging, and CI/CD integration.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training a Model](#training-a-model)
  - [Making Predictions](#making-predictions)
  - [Advanced Options](#advanced-options)
- [API Reference](#api-reference)
- [Development](#development)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality
- **Modular Architecture**: Clean separation of concerns with dedicated modules for data loading, model definition, training, evaluation, and visualization
- **Configurable**: Easily adjust hyperparameters, model architecture, and training settings
- **Command-Line Interface**: User-friendly CLI for training and prediction
- **Model Persistence**: Save and load trained models for inference
- **Comprehensive Logging**: Detailed logging throughout the pipeline for debugging and monitoring

### Model Capabilities
- **Multilayer Perceptron (MLP)**: Fully connected neural network with customizable hidden layers
- **Dropout Regularization**: Prevent overfitting with configurable dropout rates
- **Advanced Callbacks**: Model checkpointing, learning rate reduction, early stopping, and CSV logging
- **Multiple Optimizers**: Support for Adam, SGD, and other Keras optimizers

### Evaluation & Visualization
- **Rich Metrics**: Accuracy, loss, confusion matrix, per-class accuracy, top-k accuracy
- **Comprehensive Visualizations**: Training curves, confusion matrices, sample predictions, misclassified samples
- **Classification Reports**: Detailed per-class precision, recall, and F1-scores
- **Automated Reporting**: Generate summary reports with all metrics and configurations

### Engineering Best Practices
- **Type Hints**: Full type annotations for better IDE support and code quality
- **Docstrings**: Comprehensive documentation for all classes and methods
- **Unit Tests**: Extensive test suite with pytest
- **CI/CD**: Automated testing and linting with GitHub Actions
- **PEP 8 Compliant**: Clean, readable code following Python standards

## Project Structure

```
Handwritten-Digit-Classification/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── src/
│   └── mnist_classifier/
│       ├── __init__.py            # Package initialization
│       ├── config.py              # Configuration management
│       ├── data_loader.py         # Data loading and preprocessing
│       ├── model.py               # Neural network model definition
│       ├── trainer.py             # Training logic and callbacks
│       ├── evaluator.py           # Model evaluation and metrics
│       ├── visualization.py       # Plotting and visualization utilities
│       └── cli.py                 # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_config.py             # Configuration tests
│   ├── test_data_loader.py        # Data loading tests
│   ├── test_model.py              # Model tests
│   └── test_trainer.py            # Training tests
├── models/                        # Saved models (gitignored)
├── results/
│   ├── plots/                     # Generated visualizations
│   └── logs/                      # Training logs and history
├── data/                          # Dataset cache (gitignored)
├── train.py                       # Main training script
├── predict.py                     # Prediction script
├── requirements.txt               # Production dependencies
├── requirements-dev.txt           # Development dependencies
├── setup.py                       # Package installation script
├── pytest.ini                     # Pytest configuration
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pyenthusiasts/Handwritten-Digit-Classification.git
   cd Handwritten-Digit-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Development Installation

For development with testing and code quality tools:

```bash
pip install -r requirements-dev.txt
```

### Package Installation

Install as a Python package:

```bash
pip install -e .
```

This enables you to use the package from anywhere and provides command-line entry points.

## Quick Start

### Train a Model with Default Settings

```bash
python train.py
```

This will:
- Load and preprocess the MNIST dataset
- Build a neural network with default architecture (128, 64 hidden units)
- Train for 10 epochs with batch size 128
- Save the trained model to `models/`
- Generate visualizations in `results/plots/`
- Save training logs to `results/logs/`

### Make Predictions

```bash
python predict.py --model-path models/best_model.h5 --num-samples 10
```

## Usage

### Training a Model

#### Basic Training

```bash
python train.py
```

#### Custom Hyperparameters

```bash
python train.py \
    --epochs 20 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --hidden-units 256 128 64 \
    --dropout-rate 0.3
```

#### Enable Early Stopping

```bash
python train.py --early-stopping
```

#### Verbose Output

```bash
python train.py --verbose
```

### Making Predictions

#### Predict Random Samples

```bash
python predict.py --model-path models/best_model.h5 --num-samples 20
```

#### Show Only Misclassified Samples

```bash
python predict.py \
    --model-path models/best_model.h5 \
    --show-misclassified \
    --num-samples 10
```

#### Save Prediction Visualizations

```bash
python predict.py \
    --model-path models/best_model.h5 \
    --save-predictions \
    --output-dir results/predictions
```

### Advanced Options

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--batch-size` | Batch size for training | 128 |
| `--learning-rate` | Learning rate for optimizer | 0.001 |
| `--validation-split` | Fraction of data for validation | 0.2 |
| `--hidden-units` | Hidden layer sizes (space-separated) | 128 64 |
| `--dropout-rate` | Dropout rate for regularization | 0.2 |
| `--early-stopping` | Enable early stopping | False |
| `--no-callbacks` | Disable training callbacks | False |
| `--no-plots` | Disable plot generation | False |
| `--model-dir` | Directory to save models | models |
| `--results-dir` | Directory to save results | results |
| `--seed` | Random seed for reproducibility | 42 |
| `--verbose` | Enable verbose output | False |

#### Prediction Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to trained model (required) | - |
| `--num-samples` | Number of samples to predict | 10 |
| `--show-misclassified` | Show only misclassified samples | False |
| `--save-predictions` | Save prediction visualizations | False |
| `--output-dir` | Directory to save outputs | results/predictions |
| `--verbose` | Enable verbose output | False |

## API Reference

### Using the Package in Python

```python
from mnist_classifier import (
    Config,
    MNISTDataLoader,
    MNISTModel,
    ModelTrainer,
    ModelEvaluator,
    Visualizer,
)

# Create configuration
config = Config(
    epochs=15,
    batch_size=256,
    hidden_units=(256, 128, 64),
    dropout_rate=0.3
)

# Load data
data_loader = MNISTDataLoader(normalize=True, categorical=True)
(X_train, y_train), (X_test, y_test) = data_loader.load_data()

# Build and compile model
model = MNISTModel(config)
model.build()
model.compile()

# Train model
trainer = ModelTrainer(model, config)
history = trainer.train(X_train, y_train)

# Evaluate model
evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(X_test, y_test)

# Visualize results
visualizer = Visualizer(config)
visualizer.plot_training_history(history)

# Save model
model.save("my_model.h5")
```

### Key Classes

#### `Config`
Configuration dataclass for all hyperparameters and settings.

#### `MNISTDataLoader`
Handles loading and preprocessing the MNIST dataset.

**Methods:**
- `load_data()`: Load and preprocess data
- `get_data_shapes()`: Get information about data dimensions

#### `MNISTModel`
Neural network model wrapper.

**Methods:**
- `build()`: Build the neural network architecture
- `compile()`: Compile the model with optimizer and loss
- `save(filepath)`: Save model to disk
- `load(filepath)`: Load model from disk
- `summary()`: Print model architecture
- `count_parameters()`: Get parameter counts

#### `ModelTrainer`
Handles model training with callbacks and history tracking.

**Methods:**
- `train(X_train, y_train)`: Train the model
- `get_history()`: Get training history
- `get_final_metrics()`: Get final epoch metrics

#### `ModelEvaluator`
Model evaluation and metrics computation.

**Methods:**
- `evaluate(X_test, y_test)`: Evaluate on test set
- `predict(X)`: Generate predictions
- `predict_classes(X)`: Predict class labels
- `get_confusion_matrix(X_test, y_test)`: Compute confusion matrix
- `get_classification_report(X_test, y_test)`: Generate detailed report
- `get_misclassified_samples(X_test, y_test)`: Find misclassified examples
- `get_top_k_accuracy(X_test, y_test, k)`: Compute top-k accuracy
- `get_per_class_accuracy(X_test, y_test)`: Compute per-class accuracy

#### `Visualizer`
Visualization utilities for plots and reports.

**Methods:**
- `plot_training_history(history)`: Plot training curves
- `plot_confusion_matrix(cm)`: Plot confusion matrix heatmap
- `plot_sample_predictions(images, true, pred)`: Plot sample predictions
- `plot_per_class_accuracy(per_class_acc)`: Plot per-class accuracy bars
- `plot_misclassified_samples(images, true, pred)`: Plot misclassified samples
- `create_summary_report(metrics)`: Generate text summary report

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/pyenthusiasts/Handwritten-Digit-Classification.git
cd Handwritten-Digit-Classification

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### Code Quality

#### Format Code with Black

```bash
black src/ tests/ train.py predict.py
```

#### Sort Imports with isort

```bash
isort src/ tests/ train.py predict.py
```

#### Lint with flake8

```bash
flake8 src/ tests/ train.py predict.py --max-line-length=120
```

#### Type Checking with mypy

```bash
mypy src/
```

## Testing

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src/mnist_classifier --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_model.py -v
```

### Run Tests in Parallel

```bash
pytest -n auto
```

## Results

### Expected Performance

With default settings, the model typically achieves:
- **Test Accuracy**: ~98%
- **Training Time**: ~2-3 minutes on CPU
- **Parameters**: ~100K trainable parameters

### Output Files

After training, you'll find:

**Models:**
- `models/best_model.h5` - Best model based on validation accuracy
- `models/final_model.h5` - Model from final epoch

**Visualizations:**
- `results/plots/training_history.png` - Training/validation curves
- `results/plots/confusion_matrix.png` - Confusion matrix heatmap
- `results/plots/per_class_accuracy.png` - Per-class accuracy bars
- `results/plots/sample_predictions.png` - Sample predictions
- `results/plots/misclassified_samples.png` - Misclassified examples

**Logs:**
- `results/logs/training_YYYYMMDD_HHMMSS.csv` - Training metrics per epoch
- `results/logs/history_YYYYMMDD_HHMMSS.json` - Complete training history
- `results/summary_report.txt` - Summary report with all metrics

## Contributing

Contributions are welcome! Here's how you can help:

### Steps to Contribute

1. **Fork the repository**
   ```bash
   # Click the 'Fork' button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/Handwritten-Digit-Classification.git
   cd Handwritten-Digit-Classification
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Ensure all tests pass: `pytest`
   - Format code: `black .` and `isort .`
   - Lint code: `flake8 .`

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Describe your changes

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all public methods
- Maintain test coverage above 80%
- Update documentation for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- **TensorFlow/Keras**: Google Brain Team
- **Community**: Thanks to all contributors and users

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{mnist_digit_classification,
  title = {MNIST Digit Classification with Neural Network},
  author = {MNIST Classifier Team},
  year = {2024},
  url = {https://github.com/pyenthusiasts/Handwritten-Digit-Classification}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/pyenthusiasts/Handwritten-Digit-Classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pyenthusiasts/Handwritten-Digit-Classification/discussions)

---

**Made with passion for clean code and machine learning**

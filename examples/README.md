# MNIST Classifier Examples

This directory contains example notebooks and scripts demonstrating how to use the MNIST digit classifier.

## Available Examples

### 1. Quick Start (quickstart.ipynb)
Basic usage of the MNIST classifier including:
- Loading and preprocessing data
- Building and training a model
- Making predictions
- Visualizing results

### 2. Custom Model (custom_model.ipynb)
Advanced usage showing:
- Custom model architectures
- Hyperparameter tuning
- Callbacks and monitoring
- Model comparison

### 3. API Usage (api_usage.ipynb)
Using the REST API:
- Starting the API server
- Making prediction requests
- Batch predictions
- Integration examples

### 4. Model Export (model_export.ipynb)
Exporting models to different formats:
- ONNX export
- TensorFlow Lite conversion
- SavedModel format
- Deployment examples

## Running the Examples

### Prerequisites

```bash
# Install Jupyter
pip install jupyter notebook

# Or use JupyterLab
pip install jupyterlab
```

### Launch Jupyter

```bash
# From the project root
jupyter notebook examples/

# Or with JupyterLab
jupyter lab examples/
```

### Using Docker

```bash
# Run Jupyter in Docker
docker-compose up dev

# Access at http://localhost:8888
```

## Creating Your Own Examples

Feel free to create your own notebooks and contribute them back to the project!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

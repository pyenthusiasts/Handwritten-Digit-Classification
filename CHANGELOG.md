# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-11-13

### Added

#### Core Features
- Modular package structure with clean separation of concerns
- Configuration management system with `Config` dataclass
- Comprehensive data loading and preprocessing module
- Model definition with save/load functionality
- Advanced training module with callbacks and history tracking
- Comprehensive evaluation module with multiple metrics
- Visualization utilities with automated plot generation
- Command-line interface with argparse

#### Entry Points
- `train.py` - Main training script with full CLI support
- `predict.py` - Inference script for predictions
- `api.py` - FastAPI server for model serving

#### Testing & Quality
- Comprehensive unit test suite with pytest
- Test coverage for all core modules
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipeline
- Multi-version Python testing (3.8-3.11)
- Code quality tools integration (black, isort, flake8, mypy)

#### Deployment & Infrastructure
- Docker support with multi-stage builds
- Docker Compose for orchestration
- FastAPI REST API for model serving
- API endpoints for single and batch predictions
- Health check and model info endpoints
- Production-ready containerization

#### Development Tools
- Makefile for common development tasks
- pyproject.toml for modern Python packaging
- Pre-commit configuration
- Development and production requirements separation
- Package installation with setuptools

#### Documentation
- Comprehensive README with usage examples
- API reference documentation
- Contributing guidelines
- Code of conduct
- This changelog
- Type hints and docstrings throughout codebase

#### Model Features
- Model checkpointing (save best model)
- Early stopping capability
- Learning rate scheduling
- CSV logging for training metrics
- JSON export of training history
- Configurable model architecture
- Support for custom hyperparameters

#### Evaluation Features
- Confusion matrix computation
- Per-class accuracy metrics
- Top-k accuracy calculation
- Misclassified sample analysis
- Classification reports with precision/recall/F1
- Comprehensive performance metrics

#### Visualization Features
- Training/validation curves
- Confusion matrix heatmaps
- Per-class accuracy bar charts
- Sample prediction visualizations
- Misclassified sample plots
- Automated summary report generation
- All plots saved to files (no interactive display required)

### Changed
- Replaced monolithic script with modular architecture
- Improved code organization and maintainability
- Enhanced error handling and logging
- Better configuration management
- More flexible and extensible design

### Removed
- Old monolithic `mnist_digit_classification.py` script
- Hard-coded hyperparameters
- Interactive matplotlib displays (replaced with file saves)

### Technical Details

#### Dependencies
- TensorFlow >= 2.12.0
- Keras >= 2.12.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- scikit-learn >= 1.2.0
- FastAPI >= 0.104.0 (for API)
- Uvicorn (for API server)

#### Project Structure
```
src/mnist_classifier/  - Main package
tests/                 - Test suite
scripts/               - Utility scripts
examples/              - Example notebooks
models/                - Saved models
results/               - Training results
  plots/               - Generated visualizations
  logs/                - Training logs
```

#### CI/CD
- Automated testing on push and PR
- Code quality checks
- Multi-Python version testing
- Coverage reporting
- Package building

## [1.0.0] - Previous Version

### Initial Release
- Basic neural network implementation
- Single script for training and evaluation
- Manual hyperparameter configuration
- Interactive visualization
- Basic MNIST classification

---

## Version History

- **2.0.0** - Major refactoring with production-ready features
- **1.0.0** - Initial implementation

## Upgrade Guide

### From 1.x to 2.0

If upgrading from version 1.x:

1. **Install new dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update your code:**
   - Replace script imports with package imports:
     ```python
     # Old
     from mnist_digit_classification import ...

     # New
     from mnist_classifier import Config, MNISTModel, ...
     ```

3. **Use new CLI:**
   ```bash
   # Old
   python mnist_digit_classification.py

   # New
   python train.py
   python predict.py --model-path models/best_model.h5
   ```

4. **Configuration:**
   - Create `Config` object instead of hard-coding parameters
   - Use command-line arguments for customization

5. **Model loading:**
   - Models are now saved in `models/` directory
   - Use `MNISTModel.load()` method for loading

## Migration Notes

- All visualizations are now saved to files instead of displayed interactively
- Training history is automatically saved to JSON
- Models include both best (validation) and final versions
- Configuration is managed through dataclass instead of constants

---

For more details on any release, see the commit history or GitHub releases page.

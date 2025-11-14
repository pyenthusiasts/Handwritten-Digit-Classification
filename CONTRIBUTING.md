# Contributing to MNIST Digit Classification

First off, thank you for considering contributing to this project! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots if relevant**
* **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and explain which behavior you expected to see instead**
* **Explain why this enhancement would be useful**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** if you've added code that should be tested
4. **Ensure the test suite passes** (`make test`)
5. **Format your code** (`make format`)
6. **Lint your code** (`make lint`)
7. **Write a good commit message**
8. **Submit your pull request**

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pyenthusiasts/Handwritten-Digit-Classification.git
cd Handwritten-Digit-Classification
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
make install-dev
```

This will install all dependencies and set up pre-commit hooks.

### 4. Run Tests

```bash
make test
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:
- Maximum line length: 120 characters
- Use Black for code formatting
- Use isort for import sorting

### Code Formatting

Before committing, format your code:

```bash
make format
```

To check formatting without making changes:

```bash
make check-format
```

### Linting

Run linters to check code quality:

```bash
make lint
```

### Type Hints

- Add type hints to all function signatures
- Use `typing` module for complex types
- Run mypy for type checking

### Documentation

- Add docstrings to all public modules, classes, and functions
- Use Google-style docstrings
- Update README.md if you change functionality
- Add inline comments for complex logic

### Testing

- Write unit tests for new functionality
- Aim for >80% code coverage
- Use pytest for testing
- Follow the existing test structure

Example test:

```python
def test_model_build():
    """Test model building."""
    config = Config()
    model = MNISTModel(config)
    keras_model = model.build()
    assert keras_model is not None
```

## Project Structure

```
src/mnist_classifier/
├── __init__.py         # Package initialization
├── config.py           # Configuration management
├── data_loader.py      # Data loading and preprocessing
├── model.py            # Model definition
├── trainer.py          # Training logic
├── evaluator.py        # Evaluation metrics
├── visualization.py    # Visualization utilities
└── cli.py              # CLI argument parsing
```

When adding new functionality:
- Consider which module it belongs to
- Keep modules focused and cohesive
- Avoid circular dependencies

## Commit Message Guidelines

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(model): add support for custom activation functions

Added parameter to MNISTModel for specifying custom activation
functions in hidden layers.

Closes #123
```

```
fix(trainer): resolve memory leak in training loop

Fixed issue where training history was not being cleared
between runs, causing memory to grow indefinitely.
```

## Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for bug fixes or new features
3. **Ensure CI passes** - all tests and linting must pass
4. **Update CHANGELOG.md** with notable changes
5. **Request review** from maintainers
6. **Address feedback** promptly and professionally

### PR Title Format

Use the same format as commit messages:

```
feat: add hyperparameter tuning script
fix: resolve issue with image preprocessing
docs: update API documentation
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No new warnings
```

## Development Workflow

### Using Make Commands

We provide a Makefile for common tasks:

```bash
make help           # Show all available commands
make install-dev    # Install development dependencies
make test           # Run tests
make test-cov       # Run tests with coverage
make format         # Format code
make lint           # Run linters
make clean          # Clean up generated files
make all            # Run format, lint, and test
```

### Pre-commit Hooks

Pre-commit hooks are automatically installed with `make install-dev`. They will:
- Format code with Black
- Sort imports with isort
- Run flake8
- Check for common issues

To run hooks manually:

```bash
pre-commit run --all-files
```

### Running Specific Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::TestMNISTModel::test_build

# Run with coverage
pytest --cov=src/mnist_classifier

# Run with verbose output
pytest -v
```

## Documentation

### Adding New Documentation

1. Update README.md for user-facing changes
2. Update docstrings for code changes
3. Add examples for new features
4. Update API reference

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of function.

    Longer description if needed, explaining what the function
    does, its purpose, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised
        TypeError: Description of when this is raised

    Example:
        >>> function_name(42, "hello")
        True
    """
    pass
```

## Getting Help

- **Documentation**: Read the README.md and code comments
- **Issues**: Check existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Reach out to maintainers

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors list
- Special mentions for significant contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MNIST Digit Classification! 🎉

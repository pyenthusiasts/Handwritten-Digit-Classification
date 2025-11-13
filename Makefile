.PHONY: help install install-dev test lint format clean train predict docker-build docker-run api benchmark

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run tests with pytest"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black and isort"
	@echo "  make clean          - Clean up generated files"
	@echo "  make train          - Train model with default settings"
	@echo "  make predict        - Make predictions with trained model"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-run     - Run training in Docker"
	@echo "  make api            - Run API server locally"
	@echo "  make benchmark      - Run benchmark tests"
	@echo "  make all            - Run format, lint, and test"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/mnist_classifier --cov-report=html --cov-report=term

test-integration:
	pytest tests/ -v -m integration

# Code quality
lint:
	flake8 src/ tests/ train.py predict.py api.py --max-line-length=120
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ train.py predict.py api.py --line-length=120
	isort src/ tests/ train.py predict.py api.py --profile=black --line-length=120

check-format:
	black --check src/ tests/ train.py predict.py api.py --line-length=120
	isort --check-only src/ tests/ train.py predict.py api.py --profile=black

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ dist/ build/

clean-models:
	rm -f models/*.h5 models/*.keras

clean-results:
	rm -rf results/plots/* results/logs/*

clean-all: clean clean-models clean-results

# Training
train:
	python train.py

train-fast:
	python train.py --epochs 5 --batch-size 256

train-custom:
	python train.py --epochs 20 --hidden-units 256 128 64 --dropout-rate 0.3 --verbose

# Prediction
predict:
	python predict.py --model-path models/best_model.h5 --num-samples 10

predict-misclassified:
	python predict.py --model-path models/best_model.h5 --show-misclassified --num-samples 20

# Docker
docker-build:
	docker build -t mnist-classifier:latest -f Dockerfile .
	docker build -t mnist-api:latest -f Dockerfile.api .

docker-run:
	docker-compose up train

docker-api:
	docker-compose up api

docker-dev:
	docker-compose up dev

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker rmi mnist-classifier:latest mnist-api:latest 2>/dev/null || true

# API
api:
	python api.py

api-dev:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Benchmarking
benchmark:
	python scripts/benchmark.py

# Hyperparameter tuning
tune:
	python scripts/hyperparameter_tuning.py

# Model export
export-onnx:
	python scripts/export_model.py --format onnx --model-path models/best_model.h5

export-tflite:
	python scripts/export_model.py --format tflite --model-path models/best_model.h5

# Pre-commit
pre-commit:
	pre-commit run --all-files

# All checks
all: format lint test

# CI/CD simulation
ci: check-format lint test-cov
	@echo "CI checks passed!"

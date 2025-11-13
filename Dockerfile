# Multi-stage Dockerfile for MNIST Digit Classification

# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY train.py predict.py ./
COPY setup.py README.md LICENSE ./

# Create necessary directories
RUN mkdir -p models results/plots results/logs data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "train.py", "--help"]

# Labels
LABEL maintainer="MNIST Classifier Team"
LABEL version="2.0.0"
LABEL description="MNIST Digit Classification with Neural Network"

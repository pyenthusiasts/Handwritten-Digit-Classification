#!/usr/bin/env python3
"""
FastAPI server for MNIST digit classification.

Provides REST API endpoints for making predictions with trained models.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mnist_classifier import Config, MNISTModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Digit Classification API",
    description="REST API for handwritten digit recognition using neural networks",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global model variable
model = None
config = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    digit: int
    confidence: float
    probabilities: Dict[int, float]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    count: int


def load_model():
    """Load the trained model at startup."""
    global model, config

    model_path = os.getenv("MODEL_PATH", "models/best_model.h5")
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False

    try:
        logger.info(f"Loading model from {model_path}...")
        config = Config()
        model = MNISTModel(config)
        model.load(model_path)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def preprocess_image(image_data: bytes) -> np.ndarray:
    """Preprocess uploaded image for prediction.

    Args:
        image_data: Raw image bytes

    Returns:
        Preprocessed image array

    Raises:
        ValueError: If image processing fails
    """
    try:
        # Open image
        image = Image.open(BytesIO(image_data))

        # Convert to grayscale
        image = image.convert('L')

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(image, dtype='float32')

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28)

        return img_array

    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    success = load_model()
    if not success:
        logger.warning("Server started but model is not loaded!")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MNIST Digit Classification API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="2.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict digit from uploaded image.

    Args:
        file: Uploaded image file (28x28 grayscale or will be converted)

    Returns:
        Prediction with digit, confidence, and probabilities

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and preprocess image
        image_data = await file.read()
        img_array = preprocess_image(image_data)

        # Make prediction
        predictions = model.get_model().predict(img_array, verbose=0)[0]

        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        # Get all probabilities
        probabilities = {int(i): float(p) for i, p in enumerate(predictions)}

        logger.info(f"Prediction: {predicted_digit} (confidence: {confidence:.4f})")

        return PredictionResponse(
            digit=predicted_digit,
            confidence=confidence,
            probabilities=probabilities
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict digits from multiple uploaded images.

    Args:
        files: List of uploaded image files

    Returns:
        Batch prediction results

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 100:
        raise HTTPException(
            status_code=400,
            detail="Too many images. Maximum 100 images per batch."
        )

    try:
        predictions_list = []

        for file in files:
            # Read and preprocess image
            image_data = await file.read()
            img_array = preprocess_image(image_data)

            # Make prediction
            predictions = model.get_model().predict(img_array, verbose=0)[0]

            # Get predicted digit and confidence
            predicted_digit = int(np.argmax(predictions))
            confidence = float(predictions[predicted_digit])

            # Get all probabilities
            probabilities = {int(i): float(p) for i, p in enumerate(predictions)}

            predictions_list.append(PredictionResponse(
                digit=predicted_digit,
                confidence=confidence,
                probabilities=probabilities
            ))

        logger.info(f"Batch prediction completed for {len(files)} images")

        return BatchPredictionResponse(
            predictions=predictions_list,
            count=len(predictions_list)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        trainable, non_trainable = model.count_parameters()

        return {
            "architecture": {
                "input_shape": config.input_shape,
                "hidden_units": config.hidden_units,
                "dropout_rate": config.dropout_rate,
                "num_classes": config.num_classes,
            },
            "parameters": {
                "trainable": trainable,
                "non_trainable": non_trainable,
                "total": trainable + non_trainable,
            },
            "optimizer": config.optimizer,
            "loss": config.loss,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))

    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

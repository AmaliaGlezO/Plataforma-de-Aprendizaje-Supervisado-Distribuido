"""
Esquemas Pydantic para validación de datos de la API.
Define modelos para requests y responses.
"""

from .training_schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingStatus,
    ModelConfig
)
from .prediction_schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest
)

__all__ = [
    "TrainingRequest",
    "TrainingResponse", 
    "TrainingStatus",
    "ModelConfig",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest"
]
"""
ENDPOINTS DE PREDICCIÓN
Este archivo maneja todas las rutas para hacer inferencias con modelos.
Funciones: predicciones individuales, por lotes, gestión de modelos activos.
"""

from fastapi import APIRouter, HTTPException
from ..schemas.prediction_schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest
from typing import List

router = APIRouter(prefix="/api/predict", tags=["prediction"])

@router.post("/single", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Hace una predicción individual"""
    pass

@router.post("/batch", response_model=List[PredictionResponse])
async def make_batch_predictions(request: BatchPredictionRequest):
    """Hace predicciones por lotes"""
    pass

@router.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Obtiene el estado de un modelo en producción"""
    pass

@router.post("/models/{model_id}/activate")
async def activate_model(model_id: str):
    """Activa un modelo para producción"""
    pass

@router.post("/models/{model_id}/deactivate")
async def deactivate_model(model_id: str):
    """Desactiva un modelo de producción"""
    pass

@router.get("/models/{model_id}/metrics")
async def get_prediction_metrics(model_id: str):
    """Obtiene métricas de predicción de un modelo"""
    pass
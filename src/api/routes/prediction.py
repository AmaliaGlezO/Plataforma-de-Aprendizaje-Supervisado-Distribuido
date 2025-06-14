"""
ENDPOINTS DE PREDICCIÓN
Este archivo maneja todas las rutas para hacer inferencias con modelos.
Funciones: predicciones individuales, por lotes, gestión de modelos activos.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict
import time
import logging
from datetime import datetime

# Esquemas
from ..schemas.prediction_schemas import (
    PredictionRequest, 
    PredictionResponse,
    BatchPredictionRequest,
    ModelStatusResponse,
    ModelMetricsResponse
)

# Servicios internos
from ...ml_engine.model_server import ModelServer
from ...utils.metrics import PredictionMetricsCollector
from ...utils.auth import get_current_user
from ...utils.storage import ModelStorage

router = APIRouter(prefix="/api/predict", tags=["prediction"])
logger = logging.getLogger("prediction_api")

# Estado de modelos en memoria (en producción usar Redis)
active_models: Dict[str, bool] = {}
metrics_collector = PredictionMetricsCollector()
model_storage = ModelStorage()

@router.post("/single", 
            response_model=PredictionResponse,
            dependencies=[Depends(get_current_user)])
async def make_prediction(request: PredictionRequest):
    """
    Realiza una predicción individual con un modelo entrenado
    
    Args:
        request: {
            "model_id": "id-del-modelo",
            "features": {
                "Temperature": 25.0,
                "Humidity": 50.0,
                ...otros features
            }
        }
    """
    try:
        start_time = time.time()
        
        # Verificar si el modelo está activo
        if not active_models.get(request.model_id, False):
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {request.model_id} no está activo"
            )

        # Cargar modelo y hacer predicción
        server = ModelServer()
        prediction = server.predict(
            model_id=request.model_id,
            data=request.features
        )
        
        # Registrar métricas
        latency = time.time() - start_time
        metrics_collector.record_prediction(
            model_id=request.model_id,
            latency=latency,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Predicción exitosa para modelo {request.model_id}")
        
        return PredictionResponse(
            model_id=request.model_id,
            prediction=prediction,
            latency_ms=round(latency * 1000, 2)
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/batch", 
            response_model=List[PredictionResponse],
            dependencies=[Depends(get_current_user)])
async def make_batch_predictions(request: BatchPredictionRequest):
    """
    Realiza predicciones por lotes usando paralelismo
    
    Args:
        request: {
            "model_id": "id-del-modelo",
            "instances": [
                {"feature1": value1, "feature2": value2},
                {...},
                ...
            ]
        }
    """
    try:
        start_time = time.time()
        
        if not active_models.get(request.model_id, False):
            raise HTTPException(
                status_code=400,
                detail=f"Modelo {request.model_id} no está activo"
            )

        server = ModelServer()
        predictions = server.batch_predict(
            model_id=request.model_id,
            instances=request.instances
        )
        
        # Registrar métricas
        batch_latency = time.time() - start_time
        avg_latency = batch_latency / max(1, len(request.instances))
        metrics_collector.record_batch(
            model_id=request.model_id,
            batch_size=len(request.instances),
            avg_latency=avg_latency
        )
        
        return [
            PredictionResponse(
                model_id=request.model_id,
                prediction=pred,
                latency_ms=round(avg_latency * 1000, 2)
            )
            for pred in predictions
        ]
        
    except Exception as e:
        logger.error(f"Error en predicción por lotes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/models/{model_id}/status", 
           response_model=ModelStatusResponse,
           dependencies=[Depends(get_current_user)])
async def get_model_status(model_id: str):
    """Obtiene el estado y metadata de un modelo"""
    try:
        # Verificar si el modelo existe
        if not model_storage.model_exists(model_id):
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado"
            )
        
        return ModelStatusResponse(
            model_id=model_id,
            is_active=active_models.get(model_id, False),
            created_at=model_storage.get_model_metadata(model_id, "created_at"),
            last_used=model_storage.get_model_metadata(model_id, "last_used"),
            performance=metrics_collector.get_model_performance(model_id)
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/models/{model_id}/activate",
            dependencies=[Depends(get_current_user)])
async def activate_model(model_id: str):
    """Activa un modelo para servir predicciones"""
    try:
        if not model_storage.model_exists(model_id):
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado"
            )
        
        # Cargar modelo en memoria
        server = ModelServer()
        server.load_model(model_storage.get_model_path(model_id), model_id)
        
        active_models[model_id] = True
        logger.info(f"Modelo {model_id} activado")
        
        return {"message": f"Modelo {model_id} activado correctamente"}
        
    except Exception as e:
        logger.error(f"Error activando modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/models/{model_id}/deactivate",
            dependencies=[Depends(get_current_user)])
async def deactivate_model(model_id: str):
    """Desactiva un modelo para liberar recursos"""
    try:
        if model_id not in active_models:
            raise HTTPException(
                status_code=400,
                detail="Modelo no está activo"
            )
        
        # Opcional: Descargar modelo de memoria
        server = ModelServer()
        server.unload_model(model_id)
        
        active_models[model_id] = False
        logger.info(f"Modelo {model_id} desactivado")
        
        return {"message": f"Modelo {model_id} desactivado correctamente"}
        
    except Exception as e:
        logger.error(f"Error desactivando modelo: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/models/{model_id}/metrics",
           response_model=ModelMetricsResponse,
           dependencies=[Depends(get_current_user)])
async def get_prediction_metrics(model_id: str):
    """Obtiene métricas de rendimiento del modelo"""
    try:
        if not model_storage.model_exists(model_id):
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado"
            )
        
        metrics = metrics_collector.get_model_metrics(model_id)
        
        return ModelMetricsResponse(
            model_id=model_id,
            **metrics
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
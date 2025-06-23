import os
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import ray
from ray import serve

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de directorios
MODELS_DIR = "/app/models"
TRAINING_RESULTS_DIR = "/app/training_results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)

# Modelos Pydantic
class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = Field(..., example="ridge", description="Nombre del modelo")
    features: List[List[float]] = Field(..., example=[[2025, 6, 1000]], description="Features para predicción")
    return_probabilities: bool = Field(False, description="Devolver probabilidades")

class TrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_name: str = Field("energia", description="Nombre del dataset")
    selected_models: Optional[List[str]] = Field(None, example=["ridge", "randomforest"])
    test_size: float = Field(0.3, ge=0.1, le=0.5)

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    predictions: List[Union[int, float]]
    probabilities: Optional[List[List[float]]] = None
    feature_count: int
    prediction_time: float

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja eventos de inicio/cierre de la aplicación"""
    # Inicialización
    try:
        if not ray.is_initialized():
            ray.init(
                address="ray-head:6379",
                ignore_reinit_error=True,
                connection_retries=10,
                connection_retry_delay_s=3
            )
            logger.info("Conectado al cluster Ray")
        
        serve.start(detached=True)
        logger.info("Ray Serve iniciado")
    except Exception as e:
        logger.error(f"Error inicializando Ray: {e}")
        raise
    
    yield
    
    # Limpieza
    serve.shutdown()
    if ray.is_initialized():
        ray.shutdown()

# Creación de la app FastAPI
app = FastAPI(
    title="Distributed ML API",
    description="API para modelos de ML distribuidos con Ray",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clase para predicción distribuida
@serve.deployment
class ModelPredictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Modelo cargado desde {model_path}")

    async def predict(self, features: List[List[float]]):
        return self.model.predict(np.array(features)).tolist()

    async def predict_proba(self, features: List[List[float]]):
        try:
            return self.model.predict_proba(np.array(features)).tolist()
        except AttributeError:
            return None

# Endpoints
@app.get("/health")
async def health_check():
    """Verifica el estado del servicio y sus dependencias"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ray_initialized": ray.is_initialized(),
        "ray_serve_running": serve.status().http_health_check_response is not None,
        "models_loaded": len([f for f in Path(MODELS_DIR).glob("*.pkl")])
    }
    return status

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint para realizar predicciones"""
    start_time = datetime.now()
    
    try:
        model_path = os.path.join(MODELS_DIR, f"{request.model_name}.pkl")
        if not os.path.exists(model_path):
            available_models = [f.stem for f in Path(MODELS_DIR).glob("*.pkl")]
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {request.model_name} no encontrado. Disponibles: {available_models}"
            )

        # Obtener o crear deployment
        try:
            predictor = serve.get_deployment(f"predictor_{request.model_name}")
        except KeyError:
            predictor = ModelPredictor.bind(model_path)
            serve.run(predictor, name=f"predictor_{request.model_name}")

        # Realizar predicción
        predictions = await predictor.predict.remote(request.features)
        
        response = {
            "model_name": request.model_name,
            "predictions": predictions,
            "feature_count": len(request.features[0]),
            "prediction_time": (datetime.now() - start_time).total_seconds()
        }

        if request.return_probabilities:
            probs = await predictor.predict_proba.remote(request.features)
            if probs is not None:
                response["probabilities"] = probs

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Inicia el entrenamiento distribuido de modelos"""
    def train_task():
        try:
            from entrenador import EntrenamientoDistribuido
            trainer = EntrenamientoDistribuido()
            results = trainer.train_models_distributed(
                selected_models=request.selected_models,
                test_size=request.test_size
            )
            
            if results:
                trainer.save_models(MODELS_DIR)
                trainer.save_results(os.path.join(TRAINING_RESULTS_DIR, "results.json"))
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")

    background_tasks.add_task(train_task)
    
    return {
        "message": "Entrenamiento iniciado en background",
        "dataset": request.dataset_name,
        "models": request.selected_models or "todos",
        "status": "started",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
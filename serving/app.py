#!/usr/bin/env python3
"""
API REST para servir modelos de predicción de déficit energético
Usando FastAPI y los modelos entrenados con Ray
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import joblib
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Energy Deficit Prediction API",
    description="API para predicción de déficit energético usando modelos distribuidos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache de modelos en memoria
model_cache = {}
MODELS_DIR = "models/"

# Modelos de datos para la API
class EnergyData(BaseModel):
    """Modelo de datos de entrada para predicción"""
    disponibilidad: float = Field(..., description="Disponibilidad de energía (MW)")
    demanda_maxima: float = Field(..., description="Demanda máxima de energía (MW)")
    afectacion: float = Field(..., description="Afectación del sistema (MW)")
    respaldo: int = Field(..., ge=0, le=1, description="Sistema de respaldo activo (0/1)")
    horario_pico: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    unidades_averia: int = Field(..., ge=0, description="Número de unidades en avería")
    unidades_mantenimiento: int = Field(..., ge=0, description="Número de unidades en mantenimiento")
    limitacion_termica: float = Field(..., description="Limitación térmica (MW)")
    motores_impacto: float = Field(..., description="Impacto de motores (MW)")
    year: int = Field(..., ge=2020, le=2030, description="Año")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")

    class Config:
        schema_extra = {
            "example": {
                "disponibilidad": 2235.0,
                "demanda_maxima": 3170.0,
                "afectacion": 1005.0,
                "respaldo": 0,
                "horario_pico": 10,
                "unidades_averia": 4,
                "unidades_mantenimiento": 0,
                "limitacion_termica": 313.0,
                "motores_impacto": 918.0,
                "year": 2022,
                "month": 12
            }
        }

class BatchEnergyData(BaseModel):
    """Modelo para predicciones en lote"""
    data: List[EnergyData] = Field(..., description="Lista de datos para predicción")

class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    model_name: str
    predicted_deficit: float
    confidence_score: Optional[float] = None
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Respuesta de predicción en lote"""
    model_name: str
    predictions: List[float]
    processing_time_ms: float
    total_samples: int

class ModelInfo(BaseModel):
    """Información del modelo"""
    name: str
    type: str
    metrics: Dict[str, float]
    trained_at: str
    file_size_kb: float
    is_loaded: bool

class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str
    timestamp: str
    models_loaded: int
    total_models: int

# Funciones utilitarias
def get_available_models() -> List[str]:
    """Obtener lista de modelos disponibles"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
    model_names = [os.path.basename(f).replace('.joblib', '') for f in model_files]
    return model_names

def load_model(model_name: str) -> Dict[str, Any]:
    """Cargar un modelo desde disco"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_name}")
    
    try:
        model_package = joblib.load(model_path)
        logger.info(f"Modelo cargado: {model_name}")
        return model_package
    except Exception as e:
        logger.error(f"Error cargando modelo {model_name}: {str(e)}")
        raise

def get_model_info(model_name: str) -> ModelInfo:
    """Obtener información detallada de un modelo"""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_name}")
    
    # Tamaño del archivo
    file_size_kb = os.path.getsize(model_path) / 1024
    
    # Cargar información del modelo
    try:
        model_package = joblib.load(model_path)
        
        return ModelInfo(
            name=model_name,
            type=model_package.get('model_type', 'unknown'),
            metrics=model_package.get('metrics', {}),
            trained_at=model_package.get('trained_at', 'unknown'),
            file_size_kb=round(file_size_kb, 2),
            is_loaded=model_name in model_cache
        )
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accediendo al modelo: {str(e)}")

def prepare_features(data: EnergyData) -> np.ndarray:
    """Preparar features para predicción"""
    feature_order = [
        'disponibilidad', 'demanda_maxima', 'afectacion', 'respaldo',
        'horario_pico', 'unidades_averia', 'unidades_mantenimiento',
        'limitacion_termica', 'motores_impacto', 'year', 'month'
    ]
    
    features = []
    for feature in feature_order:
        features.append(getattr(data, feature))
    
    return np.array(features).reshape(1, -1)

def prepare_batch_features(data_list: List[EnergyData]) -> np.ndarray:
    """Preparar features para predicción en lote"""
    feature_order = [
        'disponibilidad', 'demanda_maxima', 'afectacion', 'respaldo',
        'horario_pico', 'unidades_averia', 'unidades_mantenimiento',
        'limitacion_termica', 'motores_impacto', 'year', 'month'
    ]
    
    features_list = []
    for data in data_list:
        features = []
        for feature in feature_order:
            features.append(getattr(data, feature))
        features_list.append(features)
    
    return np.array(features_list)

# Endpoints de la API

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "Energy Deficit Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio"""
    available_models = get_available_models()
    loaded_models = len(model_cache)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=loaded_models,
        total_models=len(available_models)
    )

@app.get("/models", response_model=List[str])
async def list_models():
    """Listar todos los modelos disponibles"""
    try:
        models = get_available_models()
        logger.info(f"Modelos disponibles: {models}")
        return models
    except Exception as e:
        logger.error(f"Error listando modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error accediendo a modelos: {str(e)}")

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_details(model_name: str):
    """Obtener información detallada de un modelo específico"""
    try:
        return get_model_info(model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str):
    """Cargar un modelo en memoria para uso rápido"""
    try:
        if model_name not in get_available_models():
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")
        
        # Cargar modelo en cache
        model_package = load_model(model_name)
        model_cache[model_name] = model_package
        
        return {
            "message": f"Modelo {model_name} cargado exitosamente",
            "model_type": model_package.get('model_type', 'unknown'),
            "cached": True
        }
    except Exception as e:
        logger.error(f"Error cargando modelo {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, data: EnergyData):
    """Realizar predicción con un modelo específico"""
    start_time = datetime.now()
    
    try:
        # Verificar si el modelo existe
        if model_name not in get_available_models():
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")
        
        # Cargar modelo (desde cache o disco)
        if model_name not in model_cache:
            logger.info(f"Cargando modelo desde disco: {model_name}")
            model_package = load_model(model_name)
            model_cache[model_name] = model_package
        else:
            model_package = model_cache[model_name]
        
        # Preparar features
        features = prepare_features(data)
        
        # Aplicar scaling si es necesario
        model = model_package['model']
        scaler = model_package.get('scaler')
        
        if scaler is not None:  # Para SVM y Linear Regression
            features = scaler.transform(features)
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calcular confidence score (para modelos que lo soporten)
        confidence_score = None
        if hasattr(model, 'predict_proba'):
            try:
                # Para clasificadores, pero adaptamos para regresión
                pass
            except:
                pass
        
        logger.info(f"Predicción realizada: {model_name} -> {prediction:.2f}")
        
        return PredictionResponse(
            model_name=model_name,
            predicted_deficit=float(prediction),
            confidence_score=confidence_score,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción con {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/{model_name}/batch", response_model=BatchPredictionResponse)
async def predict_batch(model_name: str, batch_data: BatchEnergyData):
    """Realizar predicciones en lote con un modelo específico"""
    start_time = datetime.now()
    
    try:
        # Verificar si el modelo existe
        if model_name not in get_available_models():
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_name}")
        
        # Cargar modelo (desde cache o disco)
        if model_name not in model_cache:
            logger.info(f"Cargando modelo desde disco: {model_name}")
            model_package = load_model(model_name)
            model_cache[model_name] = model_package
        else:
            model_package = model_cache[model_name]
        
        # Preparar features en lote
        features = prepare_batch_features(batch_data.data)
        
        # Aplicar scaling si es necesario
        model = model_package['model']
        scaler = model_package.get('scaler')
        
        if scaler is not None:
            features = scaler.transform(features)
        
        # Realizar predicciones
        predictions = model.predict(features)
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Predicciones en lote: {model_name} -> {len(predictions)} muestras")
        
        return BatchPredictionResponse(
            model_name=model_name,
            predictions=[float(p) for p in predictions],
            processing_time_ms=round(processing_time, 2),
            total_samples=len(predictions)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción batch con {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

@app.delete("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Descargar un modelo de la memoria"""
    if model_name in model_cache:
        del model_cache[model_name]
        return {"message": f"Modelo {model_name} descargado de memoria"}
    else:
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no está cargado en memoria")

@app.get("/cache/status")
async def cache_status():
    """Estado del cache de modelos"""
    return {
        "cached_models": list(model_cache.keys()),
        "cache_size": len(model_cache),
        "available_models": get_available_models()
    }

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Eventos de inicio de la aplicación"""
    logger.info("🚀 Iniciando Energy Deficit Prediction API")
    logger.info(f"Directorio de modelos: {MODELS_DIR}")
    
    # Verificar directorio de modelos
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Directorio de modelos no existe: {MODELS_DIR}")
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Listar modelos disponibles
    available_models = get_available_models()
    logger.info(f"Modelos disponibles: {available_models}")

@app.on_event("shutdown")
async def shutdown_event():
    """Eventos de cierre de la aplicación"""
    logger.info("🔄 Cerrando Energy Deficit Prediction API")
    model_cache.clear()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
#!/usr/bin/env python3
"""
API REST para servir modelos de predicción de déficit energético
Desarrollado con FastAPI para máximo rendimiento y documentación automática
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import json
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos cargados en memoria (cache global)
loaded_models = {}

class PredictionInput(BaseModel):
    """Esquema de entrada para predicciones"""
    disponibilidad: float = Field(..., description="Disponibilidad de energía (MW)")
    demanda_maxima: float = Field(..., description="Demanda máxima esperada (MW)")
    afectacion: float = Field(..., description="Afectación del sistema (MW)")
    respaldo: int = Field(..., ge=0, le=1, description="Sistema de respaldo activo (0/1)")
    horario_pico: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    unidades_averia: int = Field(..., ge=0, description="Número de unidades en avería")
    unidades_mantenimiento: int = Field(..., ge=0, description="Unidades en mantenimiento")
    limitacion_termica: float = Field(..., description="Limitación térmica (MW)")
    motores_impacto: float = Field(..., description="Impacto de motores (MW)")
    year: int = Field(..., ge=2020, le=2030, description="Año")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    
    @validator('disponibilidad', 'demanda_maxima', 'afectacion', 'limitacion_termica', 'motores_impacto')
    def validate_positive_values(cls, v):
        if v < 0:
            raise ValueError('Los valores de energía deben ser positivos')
        return v
    
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

class BatchPredictionInput(BaseModel):
    """Esquema para predicciones en lote"""
    data: List[PredictionInput] = Field(..., description="Lista de inputs para predicción")
    
    @validator('data')
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError('Máximo 1000 predicciones por lote')
        return v

class PredictionOutput(BaseModel):
    """Esquema de salida para predicciones"""
    model_name: str
    predicted_deficit: float
    confidence_interval: Optional[Dict[str, float]] = None
    prediction_timestamp: str
    input_data: Dict[str, Any]
    model_info: Dict[str, Any]

class ModelInfo(BaseModel):
    """Información de un modelo disponible"""
    name: str
    type: str
    metrics: Dict[str, float]
    trained_at: str
    feature_columns: List[str]
    file_size_mb: float
    status: str

class HealthCheck(BaseModel):
    """Estado de salud de la API"""
    status: str
    timestamp: str
    models_loaded: int
    uptime_seconds: float

# Startup event para cargar modelos
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando API de Serving ML...")
    await load_all_models()
    logger.info(f"✅ API lista con {len(loaded_models)} modelos cargados")
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando API...")
    loaded_models.clear()

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Déficit Energético",
    description="API REST para servir modelos de ML entrenados con Ray",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable para tracking del tiempo de inicio
start_time = datetime.now()

async def load_all_models():
    """Cargar todos los modelos disponibles en memoria"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        logger.warning(f"Directorio de modelos no encontrado: {models_dir}")
        return
    
    model_files = list(models_dir.glob("*.joblib"))
    
    if not model_files:
        logger.warning("No se encontraron modelos entrenados")
        return
    
    for model_file in model_files:
        try:
            model_name = model_file.stem
            
            # Cargar modelo
            model_package = joblib.load(model_file)
            
            # Validar estructura del modelo
            required_keys = ['model', 'feature_columns', 'model_type', 'metrics']
            if not all(key in model_package for key in required_keys):
                logger.error(f"Modelo {model_name} tiene estructura inválida")
                continue
            
            # Agregar metadata adicional
            model_package['file_path'] = str(model_file)
            model_package['file_size_mb'] = model_file.stat().st_size / (1024 * 1024)
            model_package['loaded_at'] = datetime.now().isoformat()
            
            loaded_models[model_name] = model_package
            
            logger.info(f"✓ Modelo cargado: {model_name} ({model_package['model_type']})")
            
        except Exception as e:
            logger.error(f"Error cargando modelo {model_file}: {str(e)}")

def prepare_input_features(input_data: PredictionInput) -> np.ndarray:
    """Preparar features de entrada para predicción"""
    feature_columns = [
        'disponibilidad', 'demanda_maxima', 'afectacion', 'respaldo',
        'horario_pico', 'unidades_averia', 'unidades_mantenimiento',
        'limitacion_termica', 'motores_impacto', 'year', 'month'
    ]
    
    # Crear array con las features en el orden correcto
    features = []
    for col in feature_columns:
        features.append(getattr(input_data, col))
    
    return np.array(features).reshape(1, -1)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "API de Predicción de Déficit Energético",
        "version": "1.0.0",
        "docs": "/docs",
        "models": "/models",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Verificar estado de salud de la API"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return HealthCheck(
        status="healthy" if loaded_models else "warning",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(loaded_models),
        uptime_seconds=uptime
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Listar todos los modelos disponibles"""
    if not loaded_models:
        raise HTTPException(status_code=404, detail="No hay modelos disponibles")
    
    models_info = []
    
    for name, model_package in loaded_models.items():
        model_info = ModelInfo(
            name=name,
            type=model_package['model_type'],
            metrics=model_package['metrics'],
            trained_at=model_package.get('trained_at', 'unknown'),
            feature_columns=model_package['feature_columns'],
            file_size_mb=round(model_package['file_size_mb'], 2),
            status="loaded"
        )
        models_info.append(model_info)
    
    return models_info

@app.get("/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str):
    """Obtener información detallada de un modelo específico"""
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")
    
    model_package = loaded_models[model_name]
    
    return ModelInfo(
        name=model_name,
        type=model_package['model_type'],
        metrics=model_package['metrics'],
        trained_at=model_package.get('trained_at', 'unknown'),
        feature_columns=model_package['feature_columns'],
        file_size_mb=round(model_package['file_size_mb'], 2),
        status="loaded"
    )

@app.post("/predict/{model_name}", response_model=PredictionOutput)
async def predict_with_model(model_name: str, input_data: PredictionInput):
    """Hacer predicción con un modelo específico"""
    
    # Verificar que el modelo existe
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")
    
    try:
        model_package = loaded_models[model_name]
        model = model_package['model']
        scaler = model_package.get('scaler')
        
        # Preparar features
        features = prepare_input_features(input_data)
        
        # Aplicar escalado si es necesario
        if scaler is not None:
            features = scaler.transform(features)
        
        # Hacer predicción
        prediction = model.predict(features)[0]
        
        # Preparar respuesta
        response = PredictionOutput(
            model_name=model_name,
            predicted_deficit=round(float(prediction), 2),
            prediction_timestamp=datetime.now().isoformat(),
            input_data=input_data.dict(),
            model_info={
                "type": model_package['model_type'],
                "test_r2": model_package['metrics']['test_r2'],
                "test_mae": model_package['metrics']['test_mae']
            }
        )
        
        logger.info(f"Predicción realizada con {model_name}: {prediction:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción con {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict", response_model=PredictionOutput)
async def predict_best_model(input_data: PredictionInput):
    """Hacer predicción con el mejor modelo disponible (mayor R²)"""
    
    if not loaded_models:
        raise HTTPException(status_code=404, detail="No hay modelos disponibles")
    
    # Encontrar el mejor modelo por R²
    best_model_name = max(
        loaded_models.keys(),
        key=lambda name: loaded_models[name]['metrics']['test_r2']
    )
    
    return await predict_with_model(best_model_name, input_data)

@app.post("/predict/batch/{model_name}", response_model=List[PredictionOutput])
async def batch_predict(model_name: str, batch_input: BatchPredictionInput):
    """Realizar predicciones en lote con un modelo específico"""
    
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' no encontrado")
    
    try:
        predictions = []
        
        for input_data in batch_input.data:
            prediction = await predict_with_model(model_name, input_data)
            predictions.append(prediction)
        
        logger.info(f"Predicciones en lote completadas: {len(predictions)} predicciones")
        return predictions
        
    except Exception as e:
        logger.error(f"Error en predicción en lote: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción en lote: {str(e)}")

@app.post("/reload-models")
async def reload_models(background_tasks: BackgroundTasks):
    """Recargar todos los modelos desde disco"""
    
    def reload_task():
        loaded_models.clear()
        asyncio.run(load_all_models())
    
    background_tasks.add_task(reload_task)
    
    return {"message": "Recarga de modelos iniciada en segundo plano"}

@app.get("/stats")
async def get_api_stats():
    """Obtener estadísticas de la API"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    model_stats = {}
    for name, model_package in loaded_models.items():
        model_stats[name] = {
            "type": model_package['model_type'],
            "test_r2": model_package['metrics']['test_r2'],
            "test_mae": model_package['metrics']['test_mae'],
            "size_mb": round(model_package['file_size_mb'], 2)
        }
    
    return {
        "uptime_seconds": uptime,
        "models_loaded": len(loaded_models),
        "model_stats": model_stats,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Iniciando servidor de desarrollo...")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import ray
import os
import json
import logging
import pandas as pd
import numpy as np
from io import StringIO
from ray_cluster import (
    initialize_ray_cluster,
    get_cluster_status,
    parallel_train_models,
    create_model_store,
    get_node_resources
)
from models import get_model_configs
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Distributed ML API",
    description="API para interactuar con modelos de Machine Learning entrenados en cluster Ray",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Maneja errores de validación de Pydantic con mensajes más claros"""
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        error_details.append(f"Campo '{field}': {message}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Error de validación en los datos de entrada",
            "errors": error_details,
            "type": "validation_error"
        }
    )

trainer = None
models_cache = {}
models_directory = "models"
inference_stats_file = "inference_stats.json"
global_model_manager = None
# Variable global para almacenar datos
current_data = None
current_dataset_id = None

# Modelos Pydantic para validación
class TrainConfig(BaseModel):
    models: List[str]
    params: Optional[Dict[str, Dict[str, Any]]] = {}

class PredictionRequest(BaseModel):
    model_id: str
    features: List[float]

class HealthResponse(BaseModel):
    api: str
    ray_initialized: bool
    ray_status: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    message: str
    dataset_id: str
    shape: str
    target_distribution: Dict[str, int]

class TrainResponse(BaseModel):
    message: str
    successful: int
    failed: int
    results: List[Dict[str, Any]]

class ModelsResponse(BaseModel):
    models: List[Dict[str, Any]]
    count: int

class PredictionResponse(BaseModel):
    prediction: List[float]
    probabilities: Optional[List[List[float]]] = None
    model_id: str

class ResourcesResponse(BaseModel):
    cluster_status: Dict[str, Any]
    node_resources: Dict[str, Any]

def init_ray_connection():
    """Inicializa conexión con el cluster Ray"""
    try:
        initialize_ray_cluster()
        logger.info("✅ Conexión con Ray establecida")
        return True
    except Exception as e:
        logger.error(f"❌ Error conectando a Ray: {str(e)}")
        return False

def verify_ray_connection():
    """Dependency para verificar conexión con Ray"""
    if not ray.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Ray no está inicializado"
        )
    return True


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica que la API y Ray estén funcionando"""
    ray_ok = ray.is_initialized()
    return HealthResponse(
        api="ok",
        ray_initialized=ray_ok,
        ray_status=get_cluster_status() if ray_ok else None
    )

@app.post("/upload_data", response_model=UploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """Recibe datos del frontend y los valida"""
    global current_data, current_dataset_id
    
    try:
        # Verificar que sea un archivo CSV
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Solo se aceptan archivos CSV"
            )
        
        # Leer contenido del archivo
        content = await file.read()
        
        # Convertir a DataFrame
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Validar datos básicos
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="El dataset está vacío"
            )
            
        # Validar columna target
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Columna target '{target_column}' no encontrada. Columnas disponibles: {list(df.columns)}"
            )
        
        # Extraer features y target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        current_data = (X, y)
        current_dataset_id = f"dataset_{hash(df.to_string())}"
        
        return UploadResponse(
            message="Datos cargados correctamente",
            dataset_id=current_dataset_id,
            shape=f"{X.shape[0]} filas, {X.shape[1]} features",
            target_distribution=pd.Series(y).value_counts().to_dict()
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="El archivo CSV está vacío o mal formateado"
        )
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando archivo: {str(e)}"
        )

@app.post("/train", response_model=TrainResponse)
async def train_models(
    config: TrainConfig,
    ray_connected: bool = Depends(verify_ray_connection)
):
    """Lanza entrenamiento distribuido de múltiples modelos"""
    global current_data, current_dataset_id
    
    if current_data is None:
        raise HTTPException(
            status_code=400,
            detail="No hay datos cargados. Primero sube un dataset."
        )
    
    try:
        # Preparar configuraciones de modelos
        model_configs = []
        available_models = get_model_configs().keys()
        
        for model_name in config.models:
            if model_name not in available_models:
                logger.warning(f"Modelo '{model_name}' no disponible")
                continue
                
            model_configs.append({
                'type': model_name,
                'params': config.params.get(model_name, {})
            })
        
        if not model_configs:
            raise HTTPException(
                status_code=400,
                detail=f"No hay modelos válidos para entrenar. Disponibles: {list(available_models)}"
            )
        
        # Lanzar entrenamiento paralelo
        results = parallel_train_models(current_data, model_configs, current_dataset_id)
        
        # Procesar resultados
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        return TrainResponse(
            message="Entrenamiento completado",
            successful=len(successful),
            failed=len(failed),
            results=[{
                'model_id': r.get('model_id'),
                'model_type': r.get('model_type'),
                'status': r.get('status'),
                'metrics': r.get('metrics', {}),
                'error': r.get('error')
            } for r in results]
        )
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {str(e)}"
        )

@app.get("/models", response_model=ModelsResponse)
async def get_models(ray_connected: bool = Depends(verify_ray_connection)):
    """Obtiene lista de modelos entrenados"""
    try:
        model_store = create_model_store()
        models = ray.get(model_store.get_all_models.remote())
        
        return ModelsResponse(
            models=models,
            count=len(models)
        )
    except Exception as e:
        logger.error(f"Error obteniendo modelos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo modelos: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    ray_connected: bool = Depends(verify_ray_connection)
):
    """Hace predicciones usando modelos entrenados"""
    try:
        # Obtener modelo
        model_store = create_model_store()
        model = ray.get(model_store.get_model.remote(request.model_id))
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail="Modelo no encontrado"
            )
        
        # Convertir features a formato adecuado
        features = np.array(request.features).reshape(1, -1)
        
        # Hacer predicción
        prediction = model.predict(features)
        proba = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features).tolist()
        
        return PredictionResponse(
            prediction=prediction.tolist(),
            probabilities=proba,
            model_id=request.model_id
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics(
    model_id: Optional[str] = None,
    ray_connected: bool = Depends(verify_ray_connection)
):
    """Obtiene métricas de entrenamiento y rendimiento"""
    try:
        model_store = create_model_store()
        
        if model_id:
            metrics = ray.get(model_store.get_metrics.remote(model_id))
            if not metrics:
                raise HTTPException(
                    status_code=404,
                    detail="Modelo no encontrado"
                )
            return metrics
        else:
            all_metrics = ray.get(model_store.get_metrics.remote())
            return all_metrics
            
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo métricas: {str(e)}"
        )

@app.get("/resources", response_model=ResourcesResponse)
async def get_cluster_resources(ray_connected: bool = Depends(verify_ray_connection)):
    """Obtiene estadísticas de uso de recursos del cluster"""
    try:
        return ResourcesResponse(
            cluster_status=get_cluster_status(),
            node_resources=get_node_resources()
        )
    except Exception as e:
        logger.error(f"Error obteniendo recursos: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo recursos: {str(e)}"
        )

# Endpoint adicional para obtener modelos disponibles
@app.get("/available_models")
async def get_available_models():
    """Obtiene lista de tipos de modelos disponibles"""
    try:
        available_models = get_model_configs()
        return {
            "available_models": list(available_models.keys()),
            "model_configs": available_models
        }
    except Exception as e:
        logger.error(f"Error obteniendo modelos disponibles: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo modelos disponibles: {str(e)}"
        )

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )

def main(port: int):
    """Función principal para ejecutar la API FastAPI"""
    logger.info("Iniciando API de Modelos de Machine Learning Distribuidos")
    
    os.makedirs("models", exist_ok=True)
    
    uvicorn.run(
        app,  # Aquí pasas directamente el objeto app
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
import os
import pickle
import json
import logging
import time
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import Form

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn
import ray
from ray import serve

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de directorios
BASE_DIR = Path(__file__).parent
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models_expanded'))
TRAINING_RESULTS_DIR = os.getenv('TRAINING_RESULTS_DIR', os.path.join(os.path.dirname(__file__), 'train_results'))
DATASETS_DIR = os.getenv('DATASETS_DIR', os.path.join(os.path.dirname(__file__), 'data'))

# Crear directorios si no existen
for directory in [MODELS_DIR, TRAINING_RESULTS_DIR, DATASETS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Estados globales para tracking
inference_stats = {
    "total_predictions": 0,
    "avg_prediction_time": 0.0,
    "model_usage": {},
    "last_prediction": None,
    "error_count": 0
}

# Modelos Pydantic
class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = Field(..., example="RandomForest_REG", description="Nombre del modelo")
    features: List[List[float]] = Field(..., example=[[2025, 6, 1000]], description="Features para predicción")
    return_probabilities: bool = Field(False, description="Devolver probabilidades para clasificación")

class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = Field(..., example="RandomForest_REG", description="Nombre del modelo")
    return_probabilities: bool = Field(False, description="Devolver probabilidades para clasificación")

class TrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset_name: str = Field("datos_electricos.csv", description="Nombre del dataset")
    task_type: str = Field("both", description="Tipo de tarea: 'regression', 'classification', 'both'")
    selected_models: Optional[List[str]] = Field(None, example=["RandomForest", "Ridge"])
    test_size: float = Field(0.3, ge=0.1, le=0.5, description="Proporción para test")

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    predictions: List[Union[int, float]]
    probabilities: Optional[List[List[float]]] = None
    feature_count: int
    prediction_time: float
    timestamp: str

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    file_path: str
    size_mb: float
    created_date: str
    task_type: Optional[str] = None
    algorithm: Optional[str] = None

class ClusterStatus(BaseModel):
    ray_initialized: bool
    ray_serve_running: bool
    total_nodes: int
    alive_nodes: int
    total_cpus: int
    total_memory_gb: float
    node_details: List[Dict[str, Any]]

class InferenceStats(BaseModel):
    total_predictions: int
    avg_prediction_time: float
    model_usage: Dict[str, int]
    last_prediction: Optional[str]
    error_count: int
    uptime_hours: float

def get_trainer():
    """Factory function para obtener el trainer distribuido"""
    try:
        from entrenador import EntrenamientoDistribuido
        trainer = EntrenamientoDistribuido(enable_fault_tolerance=True)
        return trainer
    except Exception as e:
        logger.error(f"Error inicializando trainer: {e}")
        raise HTTPException(status_code=500, detail=f"Error conectando al cluster: {str(e)}")

def find_model_file(model_name: str) -> Optional[str]:
    """Busca un modelo en múltiples directorios y variantes de nombre"""
    search_patterns = [
        f"{model_name}.pkl",
        f"{model_name}_REG.pkl",
        f"{model_name}_CLF.pkl",
        f"*{model_name}*.pkl"
    ]
    
    search_dirs = [MODELS_DIR, "models", "./models_expanded"]
    
    for directory in search_dirs:
        if os.path.exists(directory):
            for pattern in search_patterns:
                matches = glob.glob(os.path.join(directory, pattern))
                if matches:
                    return matches[0]
    
    return None

def update_inference_stats(model_name: str, prediction_time: float, success: bool = True):
    """Actualiza estadísticas de inferencia"""
    global inference_stats
    
    if success:
        inference_stats["total_predictions"] += 1
        # Calcular promedio móvil
        current_avg = inference_stats["avg_prediction_time"]
        total = inference_stats["total_predictions"]
        inference_stats["avg_prediction_time"] = (current_avg * (total - 1) + prediction_time) / total
        
        # Tracking por modelo
        if model_name not in inference_stats["model_usage"]:
            inference_stats["model_usage"][model_name] = 0
        inference_stats["model_usage"][model_name] += 1
        
        inference_stats["last_prediction"] = datetime.now().isoformat()
    else:
        inference_stats["error_count"] += 1

# Variable para tracking de inicio
app_start_time = datetime.now()

def init_ray_cluster():
    """Inicializa Ray con configuración robusta"""
    if ray.is_initialized():
        logger.info("Ray ya está inicializado")
        return True
    
    try:
        # Intentar conectar al cluster primero
        logger.info("Intentando conectar al cluster Ray...")
        ray.init(
            address="ray://ray-head:10001",  # Usar el puerto correcto para Ray client
            ignore_reinit_error=True,
            namespace="ml_api"
        )
        logger.info(f"✅ Conectado al cluster Ray. Nodos: {len(ray.nodes())}")
        return True
        
    except Exception as cluster_error:
        logger.warning(f"No se pudo conectar al cluster: {cluster_error}")
        
        try:
            # Fallback a modo local
            logger.info("Iniciando Ray en modo local...")
            ray.init(
                ignore_reinit_error=True,
                num_cpus=os.cpu_count(),
                namespace="ml_api"
            )
            logger.info("✅ Ray iniciado en modo local")
            return True
            
        except Exception as local_error:
            logger.error(f"Error iniciando Ray localmente: {local_error}")
            return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Maneja eventos de inicio/cierre de la aplicación"""
    # Inicialización
    ray_initialized = init_ray_cluster()
    
    if ray_initialized:
        try:
            # Iniciar Ray Serve con configuración corregida
            serve.start(
                detached=True,
                http_options={"host": "0.0.0.0", "port": 8000}
            )
            logger.info("✅ Ray Serve iniciado correctamente")
        except Exception as e:
            logger.error(f"Error iniciando Ray Serve: {e}")
            # Continuar sin Ray Serve si es necesario
    else:
        logger.warning("⚠️ Ray no se pudo inicializar - algunas funciones estarán limitadas")
    
    yield
    
    # Limpieza
    try:
        if ray.is_initialized():
            try:
                serve.shutdown()
                logger.info("Ray Serve cerrado")
            except:
                pass
            ray.shutdown()
            logger.info("Ray cerrado")
    except Exception as e:
        logger.error(f"Error durante limpieza: {e}")

# Creación de la app FastAPI
app = FastAPI(
    title="Distributed ML API",
    description="API completa de modelos distribuidos con Ray",
    version="2.0.0",
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


class ModelPredictor:
    def __init__(self, model_path: str):
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_path = model_path
            logger.info(f"✅ Modelo cargado desde {model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo desde {model_path}: {e}")
            raise

    async def predict(self, features: List[List[float]]):
        try:
            predictions = self.model.predict(np.array(features))
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise

    async def predict_proba(self, features: List[List[float]]):
        try:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(np.array(features))
                return probabilities.tolist()
            return None
        except Exception as e:
            logger.error(f"Error en predict_proba: {e}")
            return None

# ==================== ENDPOINTS ====================

@app.get("/health")
async def health_check():
    """Health check completo del servicio"""
    try:
        models_count = len([f for f in Path(MODELS_DIR).glob("*.pkl")])
        ray_nodes = ray.nodes() if ray.is_initialized() else []
        
        # Verificar si Ray Serve está funcionando
        ray_serve_running = False
        try:
            serve.list_deployments()
            ray_serve_running = True
        except:
            pass
        
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ray_initialized": ray.is_initialized(),
            "ray_serve_running": ray_serve_running,
            "models_loaded": models_count,
            "cluster_nodes": len(ray_nodes),
            "uptime_hours": round((datetime.now() - app_start_time).total_seconds() / 3600, 2),
            "inference_stats": {
                "total_predictions": inference_stats["total_predictions"],
                "error_count": inference_stats["error_count"]
            }
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Inicia el entrenamiento distribuido de modelos"""
    def train_task():
        try:
            trainer = get_trainer()
            logger.info(f"Iniciando entrenamiento:")
            
            results = trainer.train_models_distributed(
                task_type=request.task_type,
                selected_models=request.selected_models,
                test_size=request.test_size
            )
            
            if results:
                # Guardar modelos en el directorio correcto
                #trainer.save_models(MODELS_DIR)
                # Guardar resultados
                results_file = os.path.join(TRAINING_RESULTS_DIR, "train_results.json")
                #trainer.save_results(results_file)

                logger.info(f"resultados:: aaaa {results}")
                logger.info(f"✅ Entrenamiento completado. Modelos: {len(results)}")
                filtered_results = dict()
                
                for key, value in results.items():
                    entry = { x:y for x,y in value.items() if x != 'model'}
                    filtered_results[key] = entry
                return filtered_results
            else:
                logger.warning("⚠️ No se obtuvieron resultados del entrenamiento")
                
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}", exc_info=True)

    a =train_task()
    return {
        "message": "Entrenamiento completado",
        "results": a
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint para predicciones individuales con fallback sin Ray Serve"""
    start_time = time.time()
    
    try:
        model_path = find_model_file(request.model_name)
        if not model_path:
            available_models = [f.stem for f in Path(MODELS_DIR).glob("*.pkl")]
            raise HTTPException(
                status_code=404,
                detail=f"Modelo {request.model_name} no encontrado. Disponibles: {available_models}"
            )

        # Intentar usar Ray Serve si está disponible
        predictions = None
        probabilities = None
        
        if ray.is_initialized():
            try:
                deployment_name = f"predictor_{request.model_name.replace('/', '_')}"
                
                try:
                    # Intentar obtener deployment existente
                    predictor_handle = serve.get_deployment(deployment_name).get_handle()
                except:
                    # Crear nuevo deployment si no existe
                    predictor = ModelPredictor.bind(model_path)
                    serve.run(predictor, name=deployment_name)
                    predictor_handle = serve.get_deployment(deployment_name).get_handle()
                
                # Realizar predicción usando Ray Serve
                predictions = await predictor_handle.predict.remote(request.features)
                
                if request.return_probabilities:
                    probabilities = await predictor_handle.predict_proba.remote(request.features)
                
                logger.info(f"✅ Predicción completada usando Ray Serve")
                
            except Exception as serve_error:
                logger.warning(f"Error con Ray Serve, usando fallback local: {serve_error}")
                predictions = None
        
        # Fallback: predicción local sin Ray Serve
        if predictions is None:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                predictions = model.predict(np.array(request.features)).tolist()
                
                if request.return_probabilities and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(np.array(request.features)).tolist()
                
                logger.info(f"✅ Predicción completada usando fallback local")
                
            except Exception as local_error:
                logger.error(f"❌ Error en predicción local: {local_error}")
                raise HTTPException(status_code=500, detail=f"Error en predicción: {str(local_error)}")

        prediction_time = time.time() - start_time
        
        response = PredictionResponse(
            model_name=request.model_name,
            predictions=predictions,
            probabilities=probabilities,
            feature_count=len(request.features[0]) if request.features else 0,
            prediction_time=prediction_time,
            timestamp=datetime.now().isoformat()
        )

        # Actualizar estadísticas
        update_inference_stats(request.model_name, prediction_time, True)
        
        return response

    except HTTPException:
        update_inference_stats(request.model_name, 0, False)
        raise
    except Exception as e:
        update_inference_stats(request.model_name, 0, False)
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/predict/batch")
async def predict_batch(
    model_name: str = Form(...),
    return_probabilities: bool = Form(False),
    file: UploadFile = File(..., description="CSV file with features")
):
    """Predicción por lotes desde CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
    
    start_time = time.time()
    
    try:
        # Leer CSV
        content = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        
        # Convertir a array numpy
        features = df.values.tolist()
        
        # Crear request de predicción
        pred_request = PredictionRequest(
            model_name=model_name,
            features=features,
            return_probabilities=return_probabilities
        )
        
        # Usar el endpoint de predicción individual
        result = await predict(pred_request)
        
        # Agregar información del batch
        result_dict = result.dict()
        result_dict["batch_size"] = len(features)
        result_dict["batch_prediction_time"] = time.time() - start_time
        
        return result_dict
    
    except Exception as e:
        logger.error(f"❌ Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Lista todos los modelos disponibles"""
    models = []
    
    for model_file in Path(MODELS_DIR).glob("*.pkl"):
        try:
            stat = model_file.stat()
            size_mb = stat.st_size / (1024 * 1024)
            created_date = datetime.fromtimestamp(stat.st_ctime).isoformat()
            
            # Determinar tipo de tarea y algoritmo desde el nombre
            name = model_file.stem
            task_type = None
            algorithm = None
            
            if name.endswith('_REG'):
                task_type = "regression"
                algorithm = name.replace('_REG', '')
            elif name.endswith('_CLF'):
                task_type = "classification"  
                algorithm = name.replace('_CLF', '')
            else:
                algorithm = name
            
            models.append(ModelInfo(
                name=name,
                file_path=str(model_file),
                size_mb=round(size_mb, 2),
                created_date=created_date,
                task_type=task_type,
                algorithm=algorithm
            ))
        except Exception as e:
            logger.warning(f"⚠️ Error procesando modelo {model_file}: {e}")
    
    return sorted(models, key=lambda x: x.created_date, reverse=True)

@app.get("/models/search/{query}")
async def search_models(query: str):
    """Búsqueda de modelos por nombre"""
    all_models = await list_models()
    matching_models = [
        model for model in all_models 
        if query.lower() in model.name.lower() or 
           (model.algorithm and query.lower() in model.algorithm.lower())
    ]
    return matching_models

@app.get("/models/{model_name}")
async def get_model_details(model_name: str):
    """Obtiene detalles específicos de un modelo"""
    model_path = find_model_file(model_name)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado")
    
    # Información básica del archivo
    stat = Path(model_path).stat()
    
    # Intentar cargar el modelo para obtener más información
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        model_info = {
            "name": model_name,
            "file_path": model_path,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "model_type": type(model).__name__,
            "sklearn_pipeline": hasattr(model, 'steps'),
            "has_predict_proba": hasattr(model, 'predict_proba'),
            "usage_count": inference_stats["model_usage"].get(model_name, 0)
        }
        
        # Si es un pipeline, obtener información de los pasos
        if hasattr(model, 'steps'):
            model_info["pipeline_steps"] = [step[0] for step in model.steps]
        
        return model_info
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelo {model_name}: {e}")
        return {
            "name": model_name,
            "file_path": model_path,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "error": "No se pudo cargar el modelo para obtener detalles"
        }

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Elimina un modelo específico"""
    model_path = find_model_file(model_name)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado")
    
    try:
        os.remove(model_path)
        
        # Limpiar deployment si existe
        try:
            deployment_name = f"predictor_{model_name.replace('/', '_')}"
            serve.delete(deployment_name)
        except:
            pass
        
        return {
            "message": f"Modelo {model_name} eliminado exitosamente",
            "deleted_file": model_path,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando modelo: {str(e)}")

@app.get("/cluster/status", response_model=ClusterStatus)
async def get_cluster_status():
    """Obtiene el estado del cluster Ray"""
    
    try:
        if not ray.is_initialized():
            raise HTTPException(status_code=503, detail="Ray no está inicializado")
        
        cluster_resources = ray.cluster_resources()
        nodes = ray.nodes()
        
        alive_nodes = len([node for node in nodes if node.get('Alive', False)])
        
        node_details = []
        for node in nodes:
            node_details.append({
                "node_id": node.get('NodeID', 'unknown'),
                "alive": node.get('Alive', False),
                "address": node.get('NodeManagerAddress', 'unknown'),
                "resources": node.get('Resources', {})
            })
        
        return ClusterStatus(
            ray_initialized=True,
            ray_serve_running=True,  # Si llegamos aquí, está funcionando
            total_nodes=len(nodes),
            alive_nodes=alive_nodes,
            total_cpus=int(cluster_resources.get('CPU', 0)),
            total_memory_gb=round(cluster_resources.get('memory', 0) / (1024**3), 2),
            node_details=node_details
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado del cluster: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo estado del cluster: {str(e)}")

@app.get("/inference-stats", response_model=InferenceStats)
async def get_inference_stats():
    """Obtiene estadísticas de inferencia"""
    uptime_hours = (datetime.now() - app_start_time).total_seconds() / 3600
    
    return InferenceStats(
        total_predictions=inference_stats["total_predictions"],
        avg_prediction_time=inference_stats["avg_prediction_time"],
        model_usage=inference_stats["model_usage"],
        last_prediction=inference_stats["last_prediction"],
        error_count=inference_stats["error_count"],
        uptime_hours=round(uptime_hours, 2)
    )

@app.get("/algorithms")
async def get_available_algorithms():
    """Obtiene catálogo de algoritmos soportados"""
    try:
        from entrenador import EntrenamientoDistribuido
        trainer = EntrenamientoDistribuido()
        
        regression_models = trainer.get_regression_models()
        classification_models = trainer.get_classification_models()
        
        algorithms = {
            "regression": {
                name: {
                    "class": type(model).__name__,
                    "module": type(model).__module__,
                    "parameters": list(model.get_params().keys()) if hasattr(model, 'get_params') else []
                }
                for name, model in regression_models.items()
            },
            "classification": {
                name: {
                    "class": type(model).__name__,
                    "module": type(model).__module__,
                    "parameters": list(model.get_params().keys()) if hasattr(model, 'get_params') else []
                }
                for name, model in classification_models.items()
            }
        }
        
        return {
            "total_algorithms": len(regression_models) + len(classification_models),
            "regression_count": len(regression_models),
            "classification_count": len(classification_models),
            "algorithms": algorithms
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo algoritmos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/results")
async def get_training_results():
    """Obtiene los últimos resultados de entrenamiento"""
    results_file = os.path.join(TRAINING_RESULTS_DIR, "train_results.json")
    
    if not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail="No se encontraron resultados de entrenamiento")
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Agregar estadísticas resumidas
        summary = {
            "total_models": len(results),
            "successful_models": len([r for r in results.values() if r.get('status') == 'success']),
            "regression_models": len([r for r in results.values() if r.get('task_type') == 'regression']),
            "classification_models": len([r for r in results.values() if r.get('task_type') == 'classification']),
            "avg_training_time": np.mean([r.get('training_time', 0) for r in results.values()]),
            "results": results
        }
        
        return summary
    except Exception as e:
        logger.error(f"❌ Error leyendo resultados: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW ACTOR-BASED ENDPOINTS ====================

@app.get("/actor/models")
async def get_actor_models():
    """List all models stored in the global ModeloStore actor"""
    try:
        if not ray.is_initialized():
            raise HTTPException(status_code=503, detail="Ray cluster not initialized")
        
        modelo_store = ray.get_actor("ModeloStore")
        models = await modelo_store.listar_modelos.remote()
        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error getting actor models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/actor/models/{model_name}")
async def get_actor_model_details(model_name: str):
    """Get details and metrics for a specific model from the actor"""
    try:
        if not ray.is_initialized():
            raise HTTPException(status_code=503, detail="Ray cluster not initialized")
        
        modelo_store = ray.get_actor("ModeloStore")
        model = await modelo_store.obtener_modelo.remote(model_name)
        metrics = await modelo_store.obtener_metricas.remote(model_name)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in actor storage")
        
        return {
            "status": "success",
            "model_name": model_name,
            "metrics": metrics,
            "model_type": str(type(model)),
            "has_predict_proba": hasattr(model, 'predict_proba')
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting actor model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/actor/models/{model_name}")
async def delete_actor_model(model_name: str):
    """Delete a model from the global ModeloStore actor"""
    try:
        if not ray.is_initialized():
            raise HTTPException(status_code=503, detail="Ray cluster not initialized")
        
        # First check if model exists in actor
        modelo_store = ray.get_actor("ModeloStore")
        model = await modelo_store.obtener_modelo.remote(model_name)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in actor storage")
        
        # Also delete from disk if exists
        model_path = find_model_file(model_name)
        if model_path:
            try:
                os.remove(model_path)
            except Exception as e:
                logger.warning(f"Could not delete model file {model_path}: {e}")
        
        # Clean from actor by storing None
        await modelo_store.guardar_modelo.remote(model_name, None, None)
        
        return {
            "status": "success",
            "message": f"Model {model_name} removed from actor storage",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting actor model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/actor/stats")
async def get_actor_stats():
    """Get statistics from the global ModeloStore actor"""
    try:
        if not ray.is_initialized():
            raise HTTPException(status_code=503, detail="Ray cluster not initialized")
        
        modelo_store = ray.get_actor("ModeloStore")
        stats = await modelo_store.obtener_estadisticas.remote()
        
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting actor stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/actor/predict")
async def actor_predict(request: PredictionRequest):
    """Make predictions using models stored in the actor"""
    start_time = time.time()
    
    try:
        if not ray.is_initialized():
            raise HTTPException(status_code=503, detail="Ray cluster not initialized")
        
        modelo_store = ray.get_actor("ModeloStore")
        model = await modelo_store.obtener_modelo.remote(request.model_name)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {request.model_name} not found in actor storage"
            )
        
        # Convert features to numpy array
        features_array = np.array(request.features)
        
        # Make prediction
        predictions = model.predict(features_array).tolist()
        
        # Get probabilities if requested and available
        probabilities = None
        if request.return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array).tolist()
        
        prediction_time = time.time() - start_time
        
        # Update inference stats
        update_inference_stats(request.model_name, prediction_time, True)
        
        return PredictionResponse(
            model_name=request.model_name,
            predictions=predictions,
            probabilities=probabilities,
            feature_count=len(request.features[0]) if request.features else 0,
            prediction_time=prediction_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        update_inference_stats(request.model_name, 0, False)
        raise
    except Exception as e:
        update_inference_stats(request.model_name, 0, False)
        logger.error(f"Error in actor prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
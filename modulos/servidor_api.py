import requests
import streamlit as st
import pandas as pd
import os
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pickle
import json
import numpy as np
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import logging
import threading

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de FastAPI
app = FastAPI(
    title="API de ML Distribuido",
    description="API para entrenamiento y predicción de modelos de ML distribuido",
    version="1.0.0"
)

# Configuración de la API
API_PORT = 8000
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Modelos Pydantic para validación
class PredictionRequest(BaseModel):
    features: List[List[float]]
    include_probs: bool = False

class TrainRequest(BaseModel):
    dataset_name: str
    selected_models: List[str]
    test_size: float = 0.3

# Clase para el estado de la API
class APIState:
    def __init__(self):
        self.training_jobs = {}
        self.inference_stats = {}
        self.cluster_status = {}

api_state = APIState()

# --- Endpoints de la API ---

@app.get("/health")
async def health_check() -> Dict:
    """Verifica el estado de salud de la API"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def get_models() -> Dict:
    """Obtiene la lista de modelos disponibles"""
    models = {}
    for model_file in Path(MODELS_DIR).glob("*.pkl"):
        model_name = model_file.stem
        model_path = str(model_file)
        models[model_name] = {
            "path": model_path,
            "created_at": datetime.fromtimestamp(model_file.stat().st_ctime).isoformat(),
            "size": f"{model_file.stat().st_size / 1024:.2f} KB"
        }
    return {"models": models}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str) -> Dict:
    """Obtiene información detallada de un modelo"""
    model_path = Path(MODELS_DIR) / f"{model_name}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        model_info = {
            "model_name": model_name,
            "type": str(type(model)),
            "params": model.get_params() if hasattr(model, "get_params") else {},
            "created_at": datetime.fromtimestamp(model_path.stat().st_ctime).isoformat()
        }
        
        if hasattr(model, "feature_importances_"):
            model_info["feature_importances"] = model.feature_importances_.tolist()
        
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {str(e)}")

@app.post("/predict/{model_name}")
async def predict(
    model_name: str, 
    request: PredictionRequest
) -> Dict:
    """Realiza predicciones usando un modelo"""
    model_path = Path(MODELS_DIR) / f"{model_name}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        predictions = model.predict(request.features)
        result = {"predictions": predictions.tolist()}
        
        if request.include_probs and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(request.features)
            result["probabilities"] = probabilities.tolist()
        
        # Actualizar estadísticas de inferencia
        if model_name not in api_state.inference_stats:
            api_state.inference_stats[model_name] = {
                "total_predictions": 0,
                "avg_time": 0,
                "last_used": datetime.now().isoformat()
            }
        
        api_state.inference_stats[model_name]["total_predictions"] += len(request.features)
        api_state.inference_stats[model_name]["last_used"] = datetime.now().isoformat()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch/{model_name}")
async def predict_batch(
    model_name: str,
    file: UploadFile = File(...),
    include_probs: bool = Form(False)
) -> Dict:
    """Realiza predicciones en lote desde archivo"""
    model_path = Path(MODELS_DIR) / f"{model_name}.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    try:
        # Leer y procesar archivo
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
            features = df.values.tolist()
        elif file.filename.endswith('.json'):
            data = json.loads(contents)
            features = data.get("features", [])
        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
        
        # Realizar predicciones
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        predictions = model.predict(features)
        result = {
            "filename": file.filename,
            "predictions": predictions.tolist(),
            "num_predictions": len(predictions)
        }
        
        if include_probs and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            result["probabilities"] = probabilities.tolist()
        
        # Actualizar estadísticas
        if model_name not in api_state.inference_stats:
            api_state.inference_stats[model_name] = {
                "total_predictions": 0,
                "avg_time": 0,
                "last_used": datetime.now().isoformat()
            }
        
        api_state.inference_stats[model_name]["total_predictions"] += len(features)
        api_state.inference_stats[model_name]["last_used"] = datetime.now().isoformat()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción por lote: {str(e)}")

@app.post("/train")
async def train_models(request: TrainRequest) -> Dict:
    """Inicia entrenamiento de modelos"""
    # En una implementación real, esto delegaría al entrenador distribuido
    # Aquí simulamos una respuesta exitosa
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    api_state.training_jobs[job_id] = {
        "status": "running",
        "dataset": request.dataset_name,
        "models": request.selected_models,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }
    
    # Simular entrenamiento en segundo plano
    def simulate_training(job_id):
        import time
        for i in range(1, 101):
            time.sleep(0.1)
            api_state.training_jobs[job_id]["progress"] = i
        api_state.training_jobs[job_id]["status"] = "completed"
        api_state.training_jobs[job_id]["end_time"] = datetime.now().isoformat()
    
    threading.Thread(target=simulate_training, args=(job_id,)).start()
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": f"Entrenamiento iniciado para {len(request.selected_models)} modelos"
    }

@app.get("/train/status/{job_id}")
async def get_training_status(job_id: str) -> Dict:
    """Obtiene el estado de un trabajo de entrenamiento"""
    if job_id not in api_state.training_jobs:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return api_state.training_jobs[job_id]

@app.get("/cluster/status")
async def get_cluster_status() -> Dict:
    """Obtiene el estado del cluster"""
    # En una implementación real, esto consultaría el estado de Ray
    return {
        "nodes": 3,
        "alive_nodes": 3,
        "resources": {
            "CPU": 12,
            "GPU": 1,
            "memory_gb": 32.0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/datasets")
async def get_datasets() -> Dict:
    """Obtiene la lista de datasets disponibles"""
    datasets = {}
    for dataset_file in Path(DATASETS_DIR).glob("*.*"):
        datasets[dataset_file.stem] = {
            "path": str(dataset_file),
            "size": f"{dataset_file.stat().st_size / (1024*1024):.2f} MB",
            "created_at": datetime.fromtimestamp(dataset_file.stat().st_ctime).isoformat()
        }
    return {"datasets": datasets}

@app.get("/inference/stats")
async def get_inference_stats(model_name: Optional[str] = None) -> Dict:
    """Obtiene estadísticas de inferencia en tiempo real"""
    if model_name:
        if model_name not in api_state.inference_stats:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        return {model_name: api_state.inference_stats[model_name]}
    return api_state.inference_stats

# --- Cliente API para Streamlit ---

class ClienteAPI:
    def __init__(self, base_url: str = f"http://localhost:{API_PORT}"):
        self.base_url = base_url
    
    def verificar_estado(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}
    
    def obtener_modelos(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_info_modelo(self, nombre_modelo: str) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/models/{nombre_modelo}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predecir(self, nombre_modelo: str, caracteristicas: List[List[float]], incluir_probabilidades: bool = False) -> Dict:
        try:
            payload = {
                "features": caracteristicas,
                "include_probs": incluir_probabilidades
            }
            response = requests.post(
                f"{self.base_url}/predict/{nombre_modelo}",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predecir_lote(self, nombre_modelo: str, datos_archivo: bytes, nombre_archivo: str, incluir_probabilidades: bool = False) -> Dict:
        try:
            files = {'file': (nombre_archivo, datos_archivo)}
            response = requests.post(
                f"{self.base_url}/predict/batch/{nombre_modelo}",
                files=files,
                data={"include_probs": incluir_probabilidades}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def entrenar_modelos(self, nombre_dataset: str, modelos_seleccionados: Optional[List[str]] = None, tamanio_prueba: float = 0.3) -> Dict:
        try:
            payload = {
                "dataset_name": nombre_dataset,
                "selected_models": modelos_seleccionados or [],
                "test_size": tamanio_prueba
            }
            response = requests.post(f"{self.base_url}/train", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_estado_cluster(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/cluster/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_datasets(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/datasets")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_estadisticas_inferencia(self, nombre_modelo: str = None) -> Dict:
        try:
            url = f"{self.base_url}/inference/stats"
            if nombre_modelo:
                url += f"?model_name={nombre_modelo}"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# --- Interfaz Streamlit ---

def renderizar_pestana_api():
    st.header("🔌 Interfaz API")
    
    cliente_api = ClienteAPI()
    
    # Verificar estado de la API
    estado_api = cliente_api.verificar_estado()
    if estado_api.get("status") == "healthy":
        st.success("✅ API conectada y saludable")
    else:
        st.error(f"❌ Error conectando a la API: {estado_api.get('error', 'Desconocido')}")
        return
    
    # Mostrar estado del cluster
    estado_cluster = cliente_api.obtener_estado_cluster()
    if "error" not in estado_cluster:
        col1, col2, col3 = st.columns(3)
        col1.metric("🖥️ Nodos", estado_cluster.get("nodes", 0))
        col2.metric("⚡ CPUs", estado_cluster.get("resources", {}).get("CPU", 0))
        col3.metric("💾 Memoria (GB)", estado_cluster.get("resources", {}).get("memory_gb", 0))
    else:
        st.error(f"Error obteniendo estado del cluster: {estado_cluster['error']}")

def renderizar_pestana_explorar_modelos(cliente_api: ClienteAPI):
    st.header("🤖 Explorar Modelos Entrenados")
    
    modelos = cliente_api.obtener_modelos()
    if "error" in modelos:
        st.error(f"Error obteniendo modelos: {modelos['error']}")
        return
    
    if not modelos.get("models"):
        st.warning("No hay modelos entrenados disponibles")
        return
    
    modelo_seleccionado = st.selectbox(
        "Seleccionar modelo",
        list(modelos["models"].keys())
    )
    
    if modelo_seleccionado:
        info_modelo = cliente_api.obtener_info_modelo(modelo_seleccionado)
        if "error" in info_modelo:
            st.error(f"Error obteniendo información del modelo: {info_modelo['error']}")
        else:
            st.subheader(f"📋 Información de {modelo_seleccionado}")
            st.json(info_modelo)
            
            # Mostrar parámetros del modelo
            if info_modelo.get("params"):
                st.subheader("⚙️ Parámetros del Modelo")
                st.write(info_modelo["params"])
            
            # Mostrar importancia de características si está disponible
            if info_modelo.get("feature_importances"):
                st.subheader("📊 Importancia de Características")
                fig = px.bar(
                    x=range(len(info_modelo["feature_importances"])),
                    y=info_modelo["feature_importances"],
                    labels={"x": "Índice de Característica", "y": "Importancia"}
                )
                st.plotly_chart(fig)

def renderizar_pestana_predicciones(cliente_api: ClienteAPI):
    st.header("🔮 Realizar Predicciones")
    
    tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Lote"])
    
    with tab1:
        renderizar_prediccion_individual(cliente_api)
    
    with tab2:
        renderizar_prediccion_en_lote(cliente_api)

def renderizar_prediccion_individual(cliente_api: ClienteAPI):
    modelos = cliente_api.obtener_modelos()
    if "error" in modelos or not modelos.get("models"):
        st.error("No hay modelos disponibles para predicción")
        return
    
    modelo_seleccionado = st.selectbox(
        "Modelo para predicción",
        list(modelos["models"].keys()),
        key="modelo_pred_individual"
    )
    
    # Ejemplo de características basado en el modelo de energía
    caracteristicas = st.text_area(
        "Características (formato JSON array 2D)",
        value='[[5.1, 3.5, 1.4, 0.2]]',
        height=100
    )
    incluir_probabilidades = st.checkbox("Incluir probabilidades (si el modelo lo soporta)")
    
    if st.button("Predecir"):
        try:
            features = json.loads(caracteristicas)
            if not isinstance(features, list) or not all(isinstance(x, list) for x in features):
                raise ValueError("Formato incorrecto")
            
            resultado = cliente_api.predecir(
                modelo_seleccionado,
                features,
                incluir_probabilidades
            )
            
            if "error" in resultado:
                st.error(f"Error en predicción: {resultado['error']}")
            else:
                st.success("✅ Predicción exitosa")
                st.subheader("Resultados")
                st.write(resultado)
                
                # Visualización de resultados
                if "predictions" in resultado:
                    df = pd.DataFrame({
                        "input": [str(f) for f in features],
                        "prediction": resultado["predictions"]
                    })
                    st.dataframe(df)
                
                if "probabilities" in resultado:
                    st.subheader("Probabilidades")
                    prob_df = pd.DataFrame(resultado["probabilities"])
                    st.bar_chart(prob_df)
        except Exception as e:
            st.error(f"Error procesando características: {str(e)}")

def renderizar_prediccion_en_lote(cliente_api: ClienteAPI):
    modelos = cliente_api.obtener_modelos()
    if "error" in modelos or not modelos.get("models"):
        st.error("No hay modelos disponibles para predicción")
        return
    
    modelo_seleccionado = st.selectbox(
        "Modelo para predicción",
        list(modelos["models"].keys()),
        key="modelo_pred_lote"
    )
    
    archivo = st.file_uploader(
        "Subir archivo con datos (CSV o JSON)",
        type=["csv", "json"]
    )
    incluir_probabilidades = st.checkbox("Incluir probabilidades (si el modelo lo soporta)", key="probs_lote")
    
    if archivo and st.button("Predecir Lote"):
        resultado = cliente_api.predecir_lote(
            modelo_seleccionado,
            archivo.getvalue(),
            archivo.name,
            incluir_probabilidades
        )
        
        if "error" in resultado:
            st.error(f"Error en predicción: {resultado['error']}")
        else:
            st.success(f"✅ {resultado['num_predictions']} predicciones realizadas")
            st.download_button(
                "Descargar resultados",
                data=json.dumps(resultado, indent=2),
                file_name=f"predictions_{modelo_seleccionado}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

def renderizar_pestana_estadisticas_inferencia(cliente_api: ClienteAPI):
    st.header("📈 Estadísticas de Inferencia")
    
    stats = cliente_api.obtener_estadisticas_inferencia()
    if "error" in stats:
        st.error(f"Error obteniendo estadísticas: {stats['error']}")
        return
    
    if not stats:
        st.info("No hay estadísticas de inferencia disponibles")
        return
    
    modelo_seleccionado = st.selectbox(
        "Seleccionar modelo para detalles",
        list(stats.keys()) + ["Todos"],
        index=len(stats)
    )
    
    if modelo_seleccionado == "Todos":
        st.subheader("Estadísticas de Todos los Modelos")
        st.write(stats)
    else:
        st.subheader(f"Estadísticas para {modelo_seleccionado}")
        st.write(stats[modelo_seleccionado])
        
        # Gráfico de uso (simulado)
        usage_data = {
            "timestamp": [datetime.now().strftime("%H:%M:%S") for _ in range(10)],
            "predictions": [stats[modelo_seleccionado]["total_predictions"] - i*10 for i in range(10)]
        }
        fig = px.line(
            pd.DataFrame(usage_data),
            x="timestamp",
            y="predictions",
            title="Histórico de Predicciones (últimas 10 actualizaciones)"
        )
        st.plotly_chart(fig)

# --- Función para iniciar la API ---

def iniciar_api():
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)


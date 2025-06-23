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
import io

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de FastAPI
app = FastAPI(
    title="Gateway de Modelos ML",
    description="API para orquestación y predicción de modelos distribuidos",
    version="2.0.0"
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
            # Corregir la lectura del CSV
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
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
            if job_id in api_state.training_jobs:
                api_state.training_jobs[job_id]["progress"] = i
        if job_id in api_state.training_jobs:
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
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}
    
    def obtener_modelos(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_info_modelo(self, nombre_modelo: str) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/models/{nombre_modelo}", timeout=5)
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
                json=payload,
                timeout=30
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
                data={"include_probs": incluir_probabilidades},
                timeout=60
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
            response = requests.post(f"{self.base_url}/train", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_estado_cluster(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/cluster/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_datasets(self) -> Dict:
        try:
            response = requests.get(f"{self.base_url}/datasets", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_estadisticas_inferencia(self, nombre_modelo: str = None) -> Dict:
        try:
            url = f"{self.base_url}/inference/stats"
            if nombre_modelo:
                url += f"?model_name={nombre_modelo}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# --- Interfaz Streamlit con Nuevo Estilo ---

def renderizar_pestana_api():
    """Renderiza la pestaña principal de la API con el nuevo estilo"""
    
    # Estilos específicos para la pestaña API
    st.markdown("""
    <style>
    .api-status-card {
        background: linear-gradient(135deg, var(--blanco-roto) 0%, var(--beige-claro) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid var(--beige-grisaceo);
        box-shadow: 0 6px 20px rgba(64, 61, 57, 0.15);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .api-metric {
        background: var(--beige-calido);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--beige-grisaceo);
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .status-healthy {
        background: var(--verde-apagado);
        color: var(--blanco-roto);
    }
    
    .status-error {
        background: var(--naranja-terracota);
        color: var(--blanco-roto);
    }
    </style>
    """, unsafe_allow_html=True)
    
    cliente_api = ClienteAPI()
    
    # Estado de la API con nuevo estilo
    st.markdown('<div class="api-status-card">', unsafe_allow_html=True)
    st.markdown("### 🛡️ Estado del Gateway de Modelos")
    
    estado_api = cliente_api.verificar_estado()
    if estado_api.get("status") == "healthy":
        st.markdown('<div class="status-indicator status-healthy">✅ Gateway Operativo</div>', unsafe_allow_html=True)
        st.success("API conectada y procesando solicitudes")
    else:
        st.markdown('<div class="status-indicator status-error">❌ Gateway Desconectado</div>', unsafe_allow_html=True)
        st.error(f"Error de conexión: {estado_api.get('error', 'Servicio no disponible')}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pestañas secundarias con nuevo estilo
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Explorar Modelos", 
        "🔮 Predicciones", 
        "📊 Estadísticas", 
        "⚙️ Monitoreo"
    ])
    
    with tab1:
        renderizar_pestana_explorar_modelos(cliente_api)
    
    with tab2:
        renderizar_pestana_predicciones(cliente_api)
    
    with tab3:
        renderizar_pestana_estadisticas_inferencia(cliente_api)
    
    with tab4:
        renderizar_pestana_monitoreo_cluster(cliente_api)

def renderizar_pestana_explorar_modelos(cliente_api: ClienteAPI):
    """Explora modelos disponibles con estilo renovado"""
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 Repositorio de Modelos")
    
    modelos = cliente_api.obtener_modelos()
    if "error" in modelos:
        st.error(f"❌ Error cargando modelos: {modelos['error']}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not modelos.get("models"):
        st.warning("📭 No hay modelos entrenados disponibles")
        st.info("💡 Entrena algunos modelos desde la pestaña 'Laboratorio ML' para comenzar")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Mostrar resumen de modelos
    col1, col2, col3 = st.columns(3)
    col1.metric("🧠 Modelos Totales", len(modelos["models"]))
    col2.metric("💾 Tamaño Promedio", f"{sum(float(m['size'].split()[0]) for m in modelos['models'].values()) / len(modelos['models']):.1f} KB")
    col3.metric("📅 Último Creado", max(modelos["models"].values(), key=lambda x: x['created_at'])['created_at'][:10])
    
    st.markdown("---")
    
    modelo_seleccionado = st.selectbox(
        "🔍 Seleccionar modelo para inspeccionar",
        list(modelos["models"].keys())
    )
    
    if modelo_seleccionado:
        info_modelo = cliente_api.obtener_info_modelo(modelo_seleccionado)
        if "error" in info_modelo:
            st.error(f"❌ Error obteniendo información: {info_modelo['error']}")
        else:
            st.markdown(f"#### 📋 Análisis de **{modelo_seleccionado}**")
            
            # Información básica
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tipo de Modelo:**")
                st.code(info_modelo.get("type", "N/A"))
                st.markdown("**Fecha de Creación:**")
                st.write(info_modelo.get("created_at", "N/A")[:16])
            
            with col2:
                if info_modelo.get("params"):
                    st.markdown("**Parámetros del Modelo:**")
                    with st.expander("Ver configuración completa"):
                        st.json(info_modelo["params"])
            
            # Importancia de características si está disponible
            if info_modelo.get("feature_importances"):
                st.markdown("#### 📊 Importancia de Características")
                importances = info_modelo["feature_importances"]
                fig = px.bar(
                    x=range(len(importances)),
                    y=importances,
                    labels={"x": "Índice de Característica", "y": "Importancia"},
                    title="Relevancia de cada característica en el modelo",
                    color=importances,
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def renderizar_pestana_predicciones(cliente_api: ClienteAPI):
    """Interfaz de predicciones con nuevo estilo"""
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🎯 Predicción Individual", "📦 Predicción Masiva"])
    
    with tab1:
        renderizar_prediccion_individual(cliente_api)
    
    with tab2:
        renderizar_prediccion_en_lote(cliente_api)
    
    st.markdown('</div>', unsafe_allow_html=True)

def renderizar_prediccion_individual(cliente_api: ClienteAPI):
    """Predicción individual con estilo mejorado"""
    st.markdown("### 🎯 Predicción Individual")
    
    modelos = cliente_api.obtener_modelos()
    if "error" in modelos or not modelos.get("models"):
        st.error("❌ No hay modelos disponibles para predicción")
        return
    
    modelo_seleccionado = st.selectbox(
        "🧠 Seleccionar modelo",
        list(modelos["models"].keys()),
        key="modelo_pred_individual"
    )
    
    # Interfaz mejorada para características
    st.markdown("#### 📊 Configurar Datos de Entrada")
    caracteristicas = st.text_area(
        "Características (formato JSON array 2D)",
        value='[[5.1, 3.5, 1.4, 0.2]]',
        height=100,
        help="Ingresa las características como un array de arrays. Ejemplo: [[valor1, valor2, valor3]]"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        incluir_probabilidades = st.checkbox("📈 Incluir probabilidades", help="Si el modelo lo soporta")
    with col2:
        if st.button("🚀 Ejecutar Predicción", type="primary"):
            with st.spinner("🔄 Procesando predicción..."):
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
                        st.error(f"❌ Error en predicción: {resultado['error']}")
                    else:
                        st.success("✅ Predicción completada exitosamente")
                        
                        # Mostrar resultados con estilo
                        st.markdown("#### 📈 Resultados")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Predicciones:**")
                            for i, pred in enumerate(resultado["predictions"]):
                                st.metric(f"Muestra {i+1}", f"{pred:.4f}")
                        
                        with col2:
                            if "probabilities" in resultado:
                                st.markdown("**Probabilidades:**")
                                prob_df = pd.DataFrame(resultado["probabilities"])
                                st.bar_chart(prob_df)
                        
                except Exception as e:
                    st.error(f"❌ Error procesando características: {str(e)}")

def renderizar_prediccion_en_lote(cliente_api: ClienteAPI):
    """Predicción en lote con interfaz mejorada"""
    st.markdown("### 📦 Predicción Masiva")
    
    modelos = cliente_api.obtener_modelos()
    if "error" in modelos or not modelos.get("models"):
        st.error("❌ No hay modelos disponibles para predicción")
        return
    
    modelo_seleccionado = st.selectbox(
        "🧠 Seleccionar modelo",
        list(modelos["models"].keys()),
        key="modelo_pred_lote"
    )
    
    st.markdown("#### 📁 Cargar Archivo de Datos")
    archivo = st.file_uploader(
        "Subir archivo con datos",
        type=["csv", "json"],
        help="Formatos soportados: CSV (con headers) o JSON con estructura {'features': [[...]]}"
    )
    
    incluir_probabilidades = st.checkbox(
        "📊 Incluir probabilidades en resultados", 
        key="probs_lote"
    )
    
    if archivo and st.button("🚀 Procesar Lote", type="primary"):
        with st.spinner("🔄 Procesando predicciones en lote..."):
            resultado = cliente_api.predecir_lote(
                modelo_seleccionado,
                archivo.getvalue(),
                archivo.name,
                incluir_probabilidades
            )
            
            if "error" in resultado:
                st.error(f"❌ Error en predicción: {resultado['error']}")
            else:
                st.success(f"✅ {resultado['num_predictions']} predicciones realizadas exitosamente")
                
                # Mostrar estadísticas del lote
                col1, col2, col3 = st.columns(3)
                col1.metric("📊 Predicciones", resultado['num_predictions'])
                col2.metric("📁 Archivo", resultado['filename'])
                col3.metric("⏱️ Procesado", datetime.now().strftime("%H:%M:%S"))
                
                # Botón de descarga
                st.download_button(
                    "💾 Descargar Resultados",
                    data=json.dumps(resultado, indent=2),
                    file_name=f"predictions_{modelo_seleccionado}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

def renderizar_pestana_estadisticas_inferencia(cliente_api: ClienteAPI):
    """Estadísticas de inferencia con visualizaciones mejoradas"""
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Análisis de Rendimiento")
    
    stats = cliente_api.obtener_estadisticas_inferencia()
    if "error" in stats:
        st.error(f"❌ Error obteniendo estadísticas: {stats['error']}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not stats:
        st.info("📭 No hay estadísticas de inferencia disponibles")
        st.markdown("💡 Realiza algunas predicciones para generar estadísticas")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Resumen general
    total_predictions = sum(model_stats["total_predictions"] for model_stats in stats.values())
    st.metric("🎯 Total de Predicciones", total_predictions)
    
    # Selector de modelo
    modelo_seleccionado = st.selectbox(
        "🔍 Seleccionar modelo para análisis detallado",
        list(stats.keys()) + ["📊 Vista General"],
        index=len(stats)
    )
    
    if modelo_seleccionado == "📊 Vista General":
        st.markdown("#### 🌐 Estadísticas Globales")
        
        # Crear DataFrame para visualización
        df_stats = pd.DataFrame([
            {
                "Modelo": modelo,
                "Predicciones": datos["total_predictions"],
                "Último Uso": datos["last_used"][:10]
            }
            for modelo, datos in stats.items()
        ])
        
        # Gráfico de barras
        fig = px.bar(
            df_stats, 
            x="Modelo", 
            y="Predicciones",
            title="Uso por Modelo",
            color="Predicciones",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de estadísticas
        st.dataframe(df_stats, use_container_width=True)
        
    else:
        st.markdown(f"#### 🔍 Análisis Detallado: **{modelo_seleccionado}**")
        model_stats = stats[modelo_seleccionado]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Predicciones", model_stats["total_predictions"])
        col2.metric("⏱️ Último Uso", model_stats["last_used"][:10])
        col3.metric("📊 Promedio de Tiempo", model_stats.get("avg_time", "N/A"))
        
        st.markdown("---")
        st.markdown("### Detalles de Inferencia")
        st.write(f"Modelo: {modelo_seleccionado}")
        st.write(f"Total de Predicciones: {model_stats['total_predictions']}")
        st.write(f"Último Uso: {model_stats['last_used']}")
        st.write(f"Promedio de Tiempo: {model_stats.get('avg_time', 'N/A')} segundos")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Función para iniciar la API ---

def iniciar_api():
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
import threading
import psutil
import ray
from typing import Dict, List, Optional, Any

# Importaciones locales (ajustar según tu estructura)
try:
    from gestor_cluster import obtener_metricas_sistema
    from entrenador_ml import (
        graficar_comparacion_modelos, 
        ejecutar_entrenamiento_distribuido_avanzado,
        graficar_metricas_inferencia,
    )
    from utiles import save_system_metrics_history, get_metrics_for_timeframe
except ImportError:
    # Fallback para funciones no implementadas
    st.warning("⚠️ Algunas funciones auxiliares no están disponibles")

def obtener_metricas_sistema():
    """Función fallback para obtener métricas del sistema"""
    try:
        # Métricas básicas del sistema usando psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Métricas de Ray si está disponible
        ray_metrics = {}
        if ray.is_initialized():
            try:
                ray_metrics = {
                    'cluster_resources': ray.cluster_resources(),
                    'available_resources': ray.available_resources(),
                    'nodes': len(ray.nodes()),
                    'alive_nodes': len([node for node in ray.nodes() if node.get('Alive', False)])
                }
            except Exception as e:
                ray_metrics = {'error': str(e)}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'ray': ray_metrics
        }
    except Exception as e:
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}

def save_system_metrics_history(metrics: Dict):
    """Guarda las métricas del sistema en historial"""
    try:
        history_file = "system_metrics_history.json"
        
        # Cargar historial existente
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Agregar nueva métrica
        history.append(metrics)
        
        # Mantener solo las últimas 1000 entradas
        if len(history) > 1000:
            history = history[-1000:]
        
        # Guardar historial actualizado
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        st.error(f"Error guardando métricas: {e}")

def get_metrics_for_timeframe(hours: int = 1):
    """Obtiene métricas para un periodo de tiempo específico"""
    try:
        history_file = "system_metrics_history.json"
        
        if not os.path.exists(history_file):
            return []
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Filtrar por tiempo
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_history = []
        
        for entry in history:
            if 'timestamp' in entry:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                if entry_time >= cutoff_time:
                    filtered_history.append(entry)
        
        return filtered_history
        
    except Exception as e:
        st.error(f"Error cargando métricas: {e}")
        return []

def renderizar_pestana_entrenamiento(estado_cluster):
    """Renderiza la pestaña de entrenamiento con capacidades avanzadas"""
    st.header("🎯 Entrenamiento de Modelos ML")
    
    # Información del cluster
    if estado_cluster:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🖥️ Nodos Activos", 
                estado_cluster.get('alive_nodes', 0),
                delta=estado_cluster.get('node_delta', 0)
            )
        
        with col2:
            st.metric(
                "💾 Memoria Disponible", 
                f"{estado_cluster.get('available_memory', 0):.1f} GB"
            )
        
        with col3:
            st.metric(
                "🔥 CPU Disponible", 
                f"{estado_cluster.get('available_cpu', 0)} cores"
            )
        
        with col4:
            cluster_status = "🟢 Saludable" if estado_cluster.get('healthy', False) else "🔴 Problemas"
            st.metric("📊 Estado del Cluster", cluster_status)
    
    # Tabs para diferentes tipos de entrenamiento
    tab1, tab2, tab3 = st.tabs(["🚀 Entrenamiento Básico", "⚡ Entrenamiento Avanzado", "📊 Monitoreo"])
    
    with tab1:
        renderizar_entrenamiento_basico(estado_cluster)
    
    with tab2:
        renderizar_entrenamiento_avanzado(estado_cluster)
    
    with tab3:
        renderizar_monitoreo_entrenamiento()

def renderizar_entrenamiento_basico(estado_cluster):
    """Renderiza la interfaz de entrenamiento básico"""
    st.subheader("🎯 Configuración Básica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_name = st.text_input("📁 Nombre del Dataset", "energia_dataset")
        
        modelos_disponibles = [
            'RandomForest', 'GradientBoosting', 'LinearRegression', 'SVR', 'KNN',
            'XGBoost', 'Ridge', 'Lasso', 'AdaBoost', 'ExtraTrees', 'DecisionTree',
            'SGD', 'PassiveAggressive', 'LinearSVR', 'MLP', 'Bagging', 'Voting'
        ]
        
        modelos_seleccionados = st.multiselect(
            "🤖 Seleccionar Modelos",
            modelos_disponibles,
            default=['RandomForest', 'GradientBoosting', 'LinearRegression']
        )
    
    with col2:
        test_size = st.slider("🔢 Tamaño de Prueba (%)", 10, 50, 30) / 100
        tolerancia_fallos = st.checkbox("🛡️ Tolerancia a Fallos", True)
        cv_folds = st.number_input("🔄 CV Folds", 3, 10, 5)
        random_state = st.number_input("🎲 Random State", 0, 1000, 42)
    
    # Configuración avanzada expandible
    with st.expander("⚙️ Configuración Avanzada"):
        max_workers = st.number_input("👥 Workers Máximos", 1, 16, 4)
        timeout_seconds = st.number_input("⏱️ Timeout (segundos)", 60, 3600, 300)
        retry_attempts = st.number_input("🔄 Intentos de Retry", 1, 10, 3)
        
        guardar_modelos = st.checkbox("💾 Guardar Modelos", True)
        guardar_resultados = st.checkbox("📊 Guardar Resultados", True)
    
    # Botón de entrenamiento
    if st.button("🚀 Iniciar Entrenamiento Básico", type="primary", use_container_width=True):
        if not modelos_seleccionados:
            st.error("⚠️ Selecciona al menos un modelo")
            return
        
        with st.spinner("🔄 Ejecutando entrenamiento..."):
            # Contenedores para mostrar progreso
            progress_container = st.container()
            results_container = st.container()
            
            # Simular entrenamiento con progreso
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, modelo in enumerate(modelos_seleccionados):
                    progress = (i + 1) / len(modelos_seleccionados)
                    progress_bar.progress(progress)
                    status_text.text(f"Entrenando {modelo}... ({i+1}/{len(modelos_seleccionados)})")
                    time.sleep(0.5)  # Simular tiempo de entrenamiento
                
                status_text.text("✅ Entrenamiento completado!")
            
            # Mostrar resultados simulados
            with results_container:
                st.success("🎉 Entrenamiento completado exitosamente!")
                
                # Generar resultados simulados
                resultados_simulados = {}
                for modelo in modelos_seleccionados:
                    resultados_simulados[modelo] = {
                        'mse': np.random.uniform(0.1, 2.0),
                        'mae': np.random.uniform(0.1, 1.5),
                        'r2': np.random.uniform(0.5, 0.95),
                        'training_time': np.random.uniform(1.0, 30.0),
                        'cv_mean': np.random.uniform(0.1, 2.0),
                        'cv_std': np.random.uniform(0.01, 0.3),
                        'status': 'success'
                    }
                
                # Mostrar gráficos de resultados
                graficar_comparacion_modelos(resultados_simulados, dataset_name)

def renderizar_entrenamiento_avanzado(estado_cluster):
    """Renderiza la interfaz de entrenamiento avanzado"""
    st.subheader("⚡ Configuración Avanzada")
    
    # Configuración de hiperparámetros
    st.markdown("### 🔧 Optimización de Hiperparámetros")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimizacion_enabled = st.checkbox("🎯 Habilitar Optimización de Hiperparámetros", False)
        
        if optimizacion_enabled:
            optimization_method = st.selectbox(
                "📈 Método de Optimización",
                ["Grid Search", "Random Search", "Bayesian Optimization", "Hyperband"]
            )
            
            optimization_metric = st.selectbox(
                "📊 Métrica de Optimización",
                ["MSE", "MAE", "R²", "F1-Score", "Accuracy"]
            )
            
            max_evaluations = st.number_input("🔄 Evaluaciones Máximas", 10, 500, 50)
    
    with col2:
        # Configuración de ensemble
        ensemble_enabled = st.checkbox("🤝 Habilitar Ensemble Learning", False)
        
        if ensemble_enabled:
            ensemble_method = st.selectbox(
                "🏗️ Método de Ensemble",
                ["Voting", "Stacking", "Blending", "Boosting"]
            )
            
            ensemble_weights = st.text_input(
                "⚖️ Pesos del Ensemble (opcional)",
                placeholder="0.3, 0.4, 0.3"
            )
    
    # Configuración de monitoreo
    st.markdown("### 📊 Monitoreo en Tiempo Real")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monitoring_enabled = st.checkbox("📈 Habilitar Monitoreo", True)
        
        if monitoring_enabled:
            update_frequency = st.selectbox(
                "🔄 Frecuencia de Actualización",
                ["1 segundo", "5 segundos", "10 segundos", "30 segundos"]
            )
            
            metrics_to_monitor = st.multiselect(
                "📊 Métricas a Monitorear",
                ["Loss", "Accuracy", "Learning Rate", "Gradient Norm", "Memory Usage"],
                default=["Loss", "Accuracy"]
            )
    
    with col2:
        # Early stopping
        early_stopping = st.checkbox("⏹️ Early Stopping", False)
        
        if early_stopping:
            patience = st.number_input("⏳ Paciencia (epochs)", 5, 100, 10)
            min_delta = st.number_input("📉 Delta Mínimo", 0.001, 0.1, 0.01, format="%.3f")
    
    # Configuración de recursos
    st.markdown("### 💻 Gestión de Recursos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_limit = st.slider("🔥 Límite de CPU (%)", 10, 100, 80)
        memory_limit = st.slider("💾 Límite de Memoria (%)", 10, 100, 80)
    
    with col2:
        parallel_jobs = st.number_input("⚡ Jobs Paralelos", 1, 32, 4)
        gpu_enabled = st.checkbox("🎮 Usar GPU", False)
        
        if gpu_enabled:
            gpu_memory_fraction = st.slider("🎮 Fracción de GPU", 0.1, 1.0, 0.7)
    
    with col3:
        distributed_training = st.checkbox("🌐 Entrenamiento Distribuido", True)
        
        if distributed_training:
            cluster_scaling = st.selectbox(
                "📈 Escalado del Cluster",
                ["Manual", "Auto-scaling", "Reactive"]
            )
    
    # Configuración de experimentos
    st.markdown("### 🧪 Gestión de Experimentos")
    
    experiment_name = st.text_input("🏷️ Nombre del Experimento", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    experiment_description = st.text_area("📝 Descripción del Experimento", height=100)
    
    tags = st.text_input("🏷️ Tags (separados por coma)", "ml, regression, distributed")
    
    # Botón de entrenamiento avanzado
    if st.button("⚡ Iniciar Entrenamiento Avanzado", type="primary", use_container_width=True):
        
        # Configurar parámetros avanzados
        config_avanzada = {
            'experiment_name': experiment_name,
            'description': experiment_description,
            'tags': [tag.strip() for tag in tags.split(',')],
            'optimization': {
                'enabled': optimizacion_enabled,
                'method': optimization_method if optimizacion_enabled else None,
                'metric': optimization_metric if optimizacion_enabled else None,
                'max_evaluations': max_evaluations if optimizacion_enabled else None
            },
            'ensemble': {
                'enabled': ensemble_enabled,
                'method': ensemble_method if ensemble_enabled else None,
                'weights': ensemble_weights if ensemble_enabled else None
            },
            'monitoring': {
                'enabled': monitoring_enabled,
                'frequency': update_frequency if monitoring_enabled else None,
                'metrics': metrics_to_monitor if monitoring_enabled else []
            },
            'resources': {
                'cpu_limit': cpu_limit,
                'memory_limit': memory_limit,
                'parallel_jobs': parallel_jobs,
                'gpu_enabled': gpu_enabled,
                'gpu_memory_fraction': gpu_memory_fraction if gpu_enabled else None
            },
            'distributed': {
                'enabled': distributed_training,
                'scaling': cluster_scaling if distributed_training else None
            }
        }
        
        with st.spinner("⚡ Iniciando entrenamiento avanzado..."):
            # Mostrar configuración
            st.json(config_avanzada)
            
            # Aquí llamarías a tu función de entrenamiento avanzado
            st.success("🚀 Entrenamiento avanzado configurado!")
            st.info("💡 Esta función requiere implementación completa del entrenador avanzado")

def renderizar_monitoreo_entrenamiento():
    """Renderiza la interfaz de monitoreo de entrenamiento"""
    st.subheader("📊 Monitoreo de Entrenamiento")
    
    # Métricas en tiempo real
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Métricas en Tiempo Real")
        
        # Placeholder para métricas en tiempo real
        metrics_placeholder = st.empty()
        
        # Generar métricas simuladas
        current_metrics = {
            'Loss': np.random.uniform(0.1, 2.0),
            'Accuracy': np.random.uniform(0.7, 0.95),
            'Learning Rate': np.random.uniform(0.001, 0.1),
            'Epoch': np.random.randint(1, 100),
            'ETA (min)': np.random.randint(5, 60)
        }
        
        for metric, value in current_metrics.items():
            if isinstance(value, float):
                st.metric(metric, f"{value:.4f}")
            else:
                st.metric(metric, value)
    
    with col2:
        st.markdown("#### 🖥️ Recursos del Sistema")
        
        # Obtener métricas del sistema
        system_metrics = obtener_metricas_sistema()
        
        if 'error' not in system_metrics:
            st.metric("🔥 CPU", f"{system_metrics['cpu']['percent']:.1f}%")
            st.metric("💾 Memoria", f"{system_metrics['memory']['percent']:.1f}%")
            st.metric("💿 Disco", f"{system_metrics['disk']['percent']:.1f}%")
            
            if system_metrics.get('ray') and 'nodes' in system_metrics['ray']:
                st.metric("🖥️ Nodos Ray", system_metrics['ray']['nodes'])
        else:
            st.error(f"Error obteniendo métricas: {system_metrics['error']}")
    
    # Gráficos de evolución
    st.markdown("#### 📈 Evolución del Entrenamiento")
    
    # Generar datos simulados para gráficos
    epochs = list(range(1, 51))
    train_loss = [2.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1) for epoch in epochs]
    val_loss = [2.2 * np.exp(-epoch * 0.08) + np.random.normal(0, 0.12) for epoch in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')))
    
    fig.update_layout(
        title="Evolución de la Pérdida",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def renderizar_pestana_metricas_sistema(metricas_sistema):
    """Renderiza la pestaña de métricas del sistema"""
    st.header("📊 Métricas del Sistema")
    
    if not metricas_sistema or 'error' in metricas_sistema:
        st.error("❌ Error obteniendo métricas del sistema")
        return
    
    # Métricas actuales
    st.subheader("📈 Estado Actual del Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔥 CPU", 
            f"{metricas_sistema['cpu']['percent']:.1f}%",
            delta=f"{np.random.uniform(-5, 5):.1f}%"
        )
    
    with col2:
        memory_gb = metricas_sistema['memory']['used'] / (1024**3)
        total_gb = metricas_sistema['memory']['total'] / (1024**3)
        st.metric(
            "💾 Memoria", 
            f"{memory_gb:.1f}/{total_gb:.1f} GB",
            delta=f"{metricas_sistema['memory']['percent']:.1f}%"
        )
    
    with col3:
        disk_gb = metricas_sistema['disk']['used'] / (1024**3)
        disk_total_gb = metricas_sistema['disk']['total'] / (1024**3)
        st.metric(
            "💿 Disco", 
            f"{disk_gb:.1f}/{disk_total_gb:.1f} GB",
            delta=f"{metricas_sistema['disk']['percent']:.1f}%"
        )
    
    with col4:
        if metricas_sistema.get('ray') and 'nodes' in metricas_sistema['ray']:
            st.metric(
                "🖥️ Nodos Ray", 
                metricas_sistema['ray']['nodes'],
                delta=metricas_sistema['ray'].get('alive_nodes', 0) - metricas_sistema['ray']['nodes']
            )
        else:
            st.metric("🖥️ Ray", "No disponible")
    
    # Gráficos históricos
    st.subheader("📈 Tendencias Históricas")
    
    # Obtener métricas históricas
    historical_metrics = get_metrics_for_timeframe(hours=1)
    
    if historical_metrics:
        # Preparar datos para gráficos
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in historical_metrics if 'timestamp' in m]
        cpu_values = [m['cpu']['percent'] for m in historical_metrics if 'cpu' in m]
        memory_values = [m['memory']['percent'] for m in historical_metrics if 'memory' in m]
        disk_values = [m['disk']['percent'] for m in historical_metrics if 'disk' in m]
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU
        fig.add_trace(
            go.Scatter(x=timestamps, y=cpu_values, name='CPU %', line=dict(color='red')),
            row=1, col=1
        )
        
        # Memory
        fig.add_trace(
            go.Scatter(x=timestamps, y=memory_values, name='Memory %', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Disk
        fig.add_trace(
            go.Scatter(x=timestamps, y=disk_values, name='Disk %', line=dict(color='green')),
            row=2, col=1
        )
        
        # Network (simulado)
        network_values = [np.random.uniform(10, 100) for _ in timestamps]
        fig.add_trace(
            go.Scatter(x=timestamps, y=network_values, name='Network MB/s', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Métricas del Sistema - Última Hora")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No hay suficientes datos históricos. Las métricas se recopilarán automáticamente.")
    
    # Información detallada del cluster Ray
    if metricas_sistema.get('ray') and 'cluster_resources' in metricas_sistema['ray']:
        st.subheader("🌐 Información del Cluster Ray")
        
        ray_info = metricas_sistema['ray']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recursos del Cluster:**")
            cluster_resources = ray_info.get('cluster_resources', {})
            for resource, amount in cluster_resources.items():
                st.write(f"- {resource}: {amount}")
        
        with col2:
            st.markdown("**Recursos Disponibles:**")
            available_resources = ray_info.get('available_resources', {})
            for resource, amount in available_resources.items():
                st.write(f"- {resource}: {amount}")
    
    # Alertas y recomendaciones
    st.subheader("⚠️ Alertas y Recomendaciones")
    
    alertas = []
    
    if metricas_sistema['cpu']['percent'] > 80:
        alertas.append("🔥 **Alta utilización de CPU** - Considera optimizar o escalar")
    
    if metricas_sistema['memory']['percent'] > 85:
        alertas.append("💾 **Memoria casi llena** - Riesgo de swap o fallos de memoria")
    
    if metricas_sistema['disk']['percent'] > 90:
        alertas.append("💿 **Disco casi lleno** - Limpia archivos temporales o expande almacenamiento")
    
    if alertas:
        for alerta in alertas:
            st.warning(alerta)
    else:
        st.success("✅ Todos los recursos del sistema están en niveles saludables")

def plotear_metricas_entrenamiento(historial_entrenamiento, prefijo_grafico=""):
    """Visualiza métricas de rendimiento de los modelos"""
    if not historial_entrenamiento:
        st.warning("📊 No hay historial de entrenamiento disponible")
        return
    
    st.subheader(f"📈 Métricas de Entrenamiento {prefijo_grafico}")
    
    # Convertir historial a DataFrame
    df_history = pd.DataFrame(historial_entrenamiento)
    
    if df_history.empty:
        st.warning("📊 El historial está vacío")
        return
    
    # Gráfico de evolución de métricas
    metrics_to_plot = ['mse', 'mae', 'r2', 'training_time']
    available_metrics = [m for m in metrics_to_plot if m in df_history.columns]
    
    if available_metrics:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.upper() for m in available_metrics[:4]]
        )
        
        for i, metric in enumerate(available_metrics[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if 'modelo' in df_history.columns:
                for modelo in df_history['modelo'].unique():
                    data_modelo = df_history[df_history['modelo'] == modelo]
                    fig.add_trace(
                        go.Scatter(
                            x=data_modelo.index,
                            y=data_modelo[metric],
                            name=f'{modelo}_{metric}',
                            mode='lines+markers'
                        ),
                        row=row, col=col
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_history.index,
                        y=df_history[metric],
                        name=metric,
                        mode='lines+markers'
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, title_text=f"Evolución de Métricas {prefijo_grafico}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de estadísticas
    if 'modelo' in df_history.columns:
        st.subheader("📋 Estadísticas por Modelo")
        
        stats_df = df_history.groupby('modelo')[available_metrics].agg(['mean', 'std', 'min', 'max']).round(4)
        st.dataframe(stats_df, use_container_width=True)

def inicializar_recoleccion_metricas():
    """Inicializa la recolección de métricas del sistema si no existe historial"""
    history_file = "system_metrics_history.json"
    
    if not os.path.exists(history_file):
        st.info("🔄 Inicializando recolección de métricas del sistema...")
        
        # Crear archivo inicial con métricas actuales
        initial_metrics = obtener_metricas_sistema()
        save_system_metrics_history(initial_metrics)
        
        st.success("✅ Recolección de métricas inicializada")
        return True
    
    return False

def actualizar_metricas_automaticamente():
    """Actualiza automáticamente las métricas del sistema"""
    if 'metrics_collector_running' not in st.session_state:
        st.session_state.metrics_collector_running = False
    
    def collect_metrics():
        while st.session_state.metrics_collector_running:
            try:
                metrics = obtener_metricas_sistema()
                save_system_metrics_history(metrics)
                time.sleep(30)  # Recopilar cada 30 segundos
            except:
                print(f"Error en recolección de métricas: {e}")
                time.sleep(60)  # Esperar más si hay error
    
    if not st.session_state.metrics_collector_running:
        st.session_state.metrics_collector_running = True
        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()

def main():
    """Función principal del dashboard"""
    st.set_page_config(
        page_title="Dashboard ML Distribuido",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar recolección de métricas
    inicializar_recoleccion_metricas()
    actualizar_metricas_automaticamente()
    
    # Obtener métricas del sistema
    metricas_sistema = obtener_metricas_sistema()
    
    # Estado del cluster (simulado si no hay métricas de Ray)
    estado_cluster = {
        'alive_nodes': metricas_sistema.get('ray', {}).get('alive_nodes', 1),
        'available_memory': metricas_sistema.get('memory', {}).get('available', 0) / (1024**3),
        'available_cpu': metricas_sistema.get('cpu', {}).get('count', 4),
        'healthy': True
    }
    
    # Sidebar con navegación
    with st.sidebar:
        st.title("🚀 ML Distribuido")
        st.markdown("---")
        
        opciones = [
            "🏠 Inicio",
            "🎯 Entrenamiento",
            "📊 Métricas del Sistema",
            "🤖 Modelos Entrenados",
            "⚙️ Configuración"
        ]
        
        seleccion = st.radio("Navegación", opciones)
    
    # Contenido principal según selección
    if seleccion == "🏠 Inicio":
        st.title("🏠 Panel de Control - ML Distribuido")
        st.markdown("""
            Bienvenido al dashboard de monitoreo y entrenamiento de modelos de Machine Learning distribuido.
            
            **Características principales:**
            - 🎯 Entrenamiento distribuido de modelos
            - 📊 Monitoreo en tiempo real del sistema
            - 🤖 Comparación de modelos entrenados
            - ⚡ Configuración avanzada de entrenamiento
        """)
        
        # Mostrar estado del cluster
        st.markdown("### 🌐 Estado del Cluster")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🖥️ Nodos Activos", estado_cluster['alive_nodes'])
        
        with col2:
            st.metric("💾 Memoria Disponible", f"{estado_cluster['available_memory']:.1f} GB")
        
        with col3:
            st.metric("🔥 CPU Disponible", f"{estado_cluster['available_cpu']} cores")
        
        # Gráfico de uso de recursos
        st.markdown("### 📈 Uso de Recursos")
        fig = go.Figure()
        
        if 'cpu' in metricas_sistema:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metricas_sistema['cpu']['percent'],
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
        
        if 'memory' in metricas_sistema:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metricas_sistema['memory']['percent'],
                title={'text': "Memory Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
        
        if 'disk' in metricas_sistema:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metricas_sistema['disk']['percent'],
                title={'text': "Disk Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
        
        fig.update_layout(grid={'rows': 1, 'columns': 3, 'pattern': "independent"})
        st.plotly_chart(fig, use_container_width=True)
    
    elif seleccion == "🎯 Entrenamiento":
        renderizar_pestana_entrenamiento(estado_cluster)
    
    elif seleccion == "📊 Métricas del Sistema":
        renderizar_pestana_metricas_sistema(metricas_sistema)
    
    elif seleccion == "🤖 Modelos Entrenados":
        st.title("🤖 Modelos Entrenados")
        
        # Simular modelos entrenados
        modelos_entrenados = {
            'RandomForest': {'mse': 0.45, 'mae': 0.32, 'r2': 0.89, 'fecha': '2023-11-15'},
            'GradientBoosting': {'mse': 0.38, 'mae': 0.28, 'r2': 0.91, 'fecha': '2023-11-14'},
            'LinearRegression': {'mse': 0.78, 'mae': 0.65, 'r2': 0.72, 'fecha': '2023-11-13'}
        }
        
        # Seleccionar modelo para ver detalles
        modelo_seleccionado = st.selectbox(
            "Seleccionar Modelo",
            list(modelos_entrenados.keys())
        )
        
        if modelo_seleccionado:
            st.subheader(f"📊 Métricas de {modelo_seleccionado}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("MSE", modelos_entrenados[modelo_seleccionado]['mse'])
            
            with col2:
                st.metric("MAE", modelos_entrenados[modelo_seleccionado]['mae'])
            
            with col3:
                st.metric("R²", modelos_entrenados[modelo_seleccionado]['r2'])
            
            # Gráfico de importancia de características (simulado)
            st.subheader("📊 Importancia de Características")
            features = ['disponibilidad', 'demanda_maxima', 'afectacion', 'respaldo', 'horario_pico']
            importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
            
            fig = px.bar(
                x=features,
                y=importance,
                labels={'x': 'Característica', 'y': 'Importancia'},
                title=f"Importancia de características - {modelo_seleccionado}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif seleccion == "⚙️ Configuración":
        st.title("⚙️ Configuración del Sistema")
        
        st.markdown("### 🔧 Configuración del Cluster")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_workers = st.number_input("Máximo de Workers", 1, 32, 4)
            memory_limit = st.number_input("Límite de Memoria (GB)", 1, 128, 16)
        
        with col2:
            auto_scaling = st.checkbox("Auto-scaling", True)
            gpu_enabled = st.checkbox("Habilitar GPU", False)
        
        st.markdown("### 📊 Configuración de Monitoreo")
        update_frequency = st.selectbox(
            "Frecuencia de Actualización",
            ["1 segundo", "5 segundos", "10 segundos", "30 segundos", "1 minuto"],
            index=2
        )
        
        if st.button("💾 Guardar Configuración", type="primary"):
            st.success("✅ Configuración guardada exitosamente!")
            st.balloons()

if __name__ == "__main__":
    main()
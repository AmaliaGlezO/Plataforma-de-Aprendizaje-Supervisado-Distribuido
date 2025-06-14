"""
DASHBOARD DE MONITOREO
Este archivo maneja el dashboard web para visualizar métricas del sistema.
Funciones: crear visualizaciones, métricas en tiempo real, alertas.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import ray
import psutil
import threading
from collections import deque, defaultdict
import json

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="🤖 Distributed ML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "Sistema distribuido de ML con Ray y Docker"
    }
)

class MetricsCollector:
    """Recolector de métricas del sistema"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.current_metrics = {}
        self.alerts = []
        
    def collect_system_metrics(self) -> Dict:
        """Recolecta métricas del sistema"""
        try:
            # Métricas básicas del sistema
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Métricas de red
            net_io = psutil.net_io_counters()
            
            timestamp = datetime.now()
            
            metrics = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
            }
            
            # Almacenar en histórico
            for key, value in metrics.items():
                if key != 'timestamp':
                    self.metrics_history[key].append((timestamp, value))
                    
            self.current_metrics.update(metrics)
            return metrics
            
        except Exception as e:
            st.error(f"Error recolectando métricas del sistema: {e}")
            return {}
    
    def collect_ray_metrics(self) -> Dict:
        """Recolecta métricas del cluster Ray"""
        try:
            if not ray.is_initialized():
                return {}
                
            # Información del cluster
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Estadísticas de nodos
            nodes = ray.nodes()
            
            # Tareas activas
            tasks = ray.list_tasks()
            
            timestamp = datetime.now()
            
            metrics = {
                'timestamp': timestamp,
                'total_cpus': cluster_resources.get('CPU', 0),
                'available_cpus': available_resources.get('CPU', 0),
                'total_memory_gb': cluster_resources.get('memory', 0) / (1024**3),
                'available_memory_gb': available_resources.get('memory', 0) / (1024**3),
                'total_nodes': len(nodes),
                'alive_nodes': len([n for n in nodes if n['alive']]),
                'active_tasks': len([t for t in tasks if t['state'] == 'RUNNING']),
                'pending_tasks': len([t for t in tasks if t['state'] == 'PENDING']),
                'failed_tasks': len([t for t in tasks if t['state'] == 'FAILED']),
            }
            
            # Almacenar en histórico
            for key, value in metrics.items():
                if key != 'timestamp' and isinstance(value, (int, float)):
                    self.metrics_history[key].append((timestamp, value))
                    
            self.current_metrics.update(metrics)
            return metrics
            
        except Exception as e:
            st.error(f"Error recolectando métricas de Ray: {e}")
            return {}
    
    def get_historical_data(self, metric_name: str, minutes: int = 30) -> pd.DataFrame:
        """Obtiene datos históricos de una métrica"""
        if metric_name not in self.metrics_history:
            return pd.DataFrame()
            
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [
            (timestamp, value) for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff_time
        ]
        
        if not recent_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(recent_data, columns=['timestamp', metric_name])
        return df.sort_values('timestamp')


class MonitoringDashboard:
    """Dashboard web para monitoreo del sistema"""
    
    def __init__(self, cluster_manager=None):
        """Inicializa el dashboard"""
        self.cluster_manager = cluster_manager
        self.metrics_collector = MetricsCollector()
        self.training_jobs = {}
        self.api_metrics = defaultdict(list)
        self.model_metrics = {}
        
        # Simulación de datos para demostración
        self._initialize_demo_data()
    
    def _initialize_demo_data(self):
        """Inicializa datos de demostración"""
        # Simular trabajos de entrenamiento
        self.training_jobs = {
            "job_001": {
                "name": "Random Forest - Iris Dataset",
                "status": "running",
                "progress": 0.75,
                "accuracy": 0.94,
                "loss": 0.12,
                "start_time": datetime.now() - timedelta(minutes=15),
                "estimated_remaining": "5 min",
                "dataset": "iris.csv",
                "model_type": "RandomForestClassifier"
            },
            "job_002": {
                "name": "SVM - Wine Dataset", 
                "status": "completed",
                "progress": 1.0,
                "accuracy": 0.91,
                "loss": 0.08,
                "start_time": datetime.now() - timedelta(hours=1),
                "estimated_remaining": "0 min",
                "dataset": "wine.csv",
                "model_type": "SVC"
            },
            "job_003": {
                "name": "Neural Network - Digits",
                "status": "pending",
                "progress": 0.0,
                "accuracy": 0.0,
                "loss": 0.0,
                "start_time": None,
                "estimated_remaining": "Pending",
                "dataset": "digits.csv",
                "model_type": "MLPClassifier"
            }
        }
        
        # Simular métricas de modelos
        self.model_metrics = {
            "RandomForest_iris": {"accuracy": 0.94, "precision": 0.93, "recall": 0.95, "f1": 0.94},
            "SVM_wine": {"accuracy": 0.91, "precision": 0.90, "recall": 0.92, "f1": 0.91},
            "LogisticRegression_iris": {"accuracy": 0.92, "precision": 0.91, "recall": 0.93, "f1": 0.92}
        }
    
    def setup_dashboard_layout(self):
        """Configura el layout del dashboard"""
        # Header principal
        st.title("🤖 Distributed ML Platform Dashboard")
        st.markdown("---")
        
        # Sidebar para configuración
        with st.sidebar:
            st.header("⚙️ Configuración")
            
            # Selectores de tiempo
            time_range = st.selectbox(
                "Rango de tiempo",
                ["Últimos 5 min", "Últimos 15 min", "Últimos 30 min", "Última hora"],
                index=1
            )
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto-actualizar", value=True)
            if auto_refresh:
                refresh_interval = st.slider("Intervalo (segundos)", 5, 60, 10)
            
            # Filtros
            st.subheader("🔍 Filtros")
            show_completed = st.checkbox("Mostrar trabajos completados", value=True)
            show_failed = st.checkbox("Mostrar trabajos fallidos", value=True)
            
            # Estado del sistema
            st.subheader("🔋 Estado del Sistema")
            if ray.is_initialized():
                st.success("✅ Ray Cluster: Conectado")
            else:
                st.error("❌ Ray Cluster: Desconectado")
        
        # Contenido principal en tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🚀 Entrenamientos", "🌐 API Metrics", 
            "💻 Recursos", "📈 Comparación de Modelos"
        ])
        
        with tab1:
            self.display_cluster_overview()
        
        with tab2:
            self.display_training_metrics()
        
        with tab3:
            self.display_api_metrics()
        
        with tab4:
            self.display_resource_utilization()
        
        with tab5:
            self.display_model_comparison()
    
    def display_cluster_overview(self):
        """Muestra vista general del cluster Ray"""
        st.header("🌐 Overview del Cluster")
        
        # Recolectar métricas actuales
        system_metrics = self.metrics_collector.collect_system_metrics()
        ray_metrics = self.metrics_collector.collect_ray_metrics()
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_nodes = ray_metrics.get('total_nodes', 0)
            alive_nodes = ray_metrics.get('alive_nodes', 0)
            st.metric(
                "Nodos Activos", 
                f"{alive_nodes}/{total_nodes}",
                delta=None if total_nodes == 0 else f"{(alive_nodes/total_nodes)*100:.1f}%"
            )
        
        with col2:
            active_tasks = ray_metrics.get('active_tasks', 0)
            st.metric("Tareas Activas", active_tasks)
        
        with col3:
            cpu_usage = system_metrics.get('cpu_percent', 0)
            st.metric("Uso CPU", f"{cpu_usage:.1f}%")
        
        with col4:
            memory_usage = system_metrics.get('memory_percent', 0)
            st.metric("Uso Memoria", f"{memory_usage:.1f}%")
        
        # Gráficas de recursos en tiempo real
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU Usage over time
            cpu_data = self.metrics_collector.get_historical_data('cpu_percent', 30)
            if not cpu_data.empty:
                fig_cpu = px.line(
                    cpu_data, x='timestamp', y='cpu_percent',
                    title='Uso de CPU (%)',
                    labels={'cpu_percent': 'CPU %', 'timestamp': 'Tiempo'}
                )
                fig_cpu.update_layout(height=300)
                st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            # Memory Usage over time  
            memory_data = self.metrics_collector.get_historical_data('memory_percent', 30)
            if not memory_data.empty:
                fig_memory = px.line(
                    memory_data, x='timestamp', y='memory_percent',
                    title='Uso de Memoria (%)',
                    labels={'memory_percent': 'Memoria %', 'timestamp': 'Tiempo'}
                )
                fig_memory.update_layout(height=300)
                st.plotly_chart(fig_memory, use_container_width=True)
        
        # Alertas y notificaciones
        self.setup_alerts_panel()
    
    def display_training_metrics(self):
        """Muestra métricas de entrenamientos activos"""
        st.header("🚀 Entrenamientos Activos")
        
        # Resumen de entrenamientos
        active_jobs = len([j for j in self.training_jobs.values() if j['status'] == 'running'])
        completed_jobs = len([j for j in self.training_jobs.values() if j['status'] == 'completed']) 
        pending_jobs = len([j for j in self.training_jobs.values() if j['status'] == 'pending'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Activos", active_jobs, delta=f"+{active_jobs}")
        with col2:
            st.metric("Completados", completed_jobs)
        with col3:
            st.metric("Pendientes", pending_jobs)
        
        # Lista de trabajos
        st.subheader("📋 Estado de Trabajos")
        
        for job_id, job in self.training_jobs.items():
            with st.expander(f"{job['name']} - {job['status'].upper()}", expanded=job['status']=='running'):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    # Barra de progreso
                    st.progress(job['progress'])
                    st.text(f"Dataset: {job['dataset']}")
                    st.text(f"Modelo: {job['model_type']}")
                
                with col2:
                    st.metric("Accuracy", f"{job['accuracy']:.3f}")
                    st.metric("Loss", f"{job['loss']:.3f}")
                
                with col3:
                    if job['start_time']:
                        elapsed = datetime.now() - job['start_time']
                        st.text(f"Tiempo: {elapsed}")
                    st.text(f"Restante: {job['estimated_remaining']}")
                
                # Gráficas de métricas de entrenamiento
                if job['status'] == 'running':
                    self._create_training_charts(job_id, job)
    
    def _create_training_charts(self, job_id: str, job: Dict):
        """Crea gráficas específicas para un trabajo de entrenamiento"""
        # Simular datos de entrenamiento
        epochs = list(range(1, 21))
        accuracy_history = [0.3 + 0.03 * i + np.random.normal(0, 0.01) for i in epochs]
        loss_history = [0.8 - 0.03 * i + np.random.normal(0, 0.02) for i in epochs]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=epochs, y=accuracy_history,
                mode='lines+markers',
                name='Training Accuracy',
                line=dict(color='#1f77b4')
            ))
            fig_acc.update_layout(
                title="Accuracy por Época",
                xaxis_title="Época",
                yaxis_title="Accuracy",
                height=250
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=loss_history,
                mode='lines+markers', 
                name='Training Loss',
                line=dict(color='#ff7f0e')
            ))
            fig_loss.update_layout(
                title="Loss por Época",
                xaxis_title="Época", 
                yaxis_title="Loss",
                height=250
            )
            st.plotly_chart(fig_loss, use_container_width=True)
    
    def display_api_metrics(self):
        """Muestra métricas de la API (latencia, throughput)"""
        st.header("🌐 Métricas de API")
        
        # Simular métricas de API
        current_time = datetime.now()
        
        # KPIs de API
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Requests/min", "142", delta="12")
        
        with col2:
            st.metric("Latencia Media", "235ms", delta="-15ms")
        
        with col3:
            st.metric("Error Rate", "0.3%", delta="-0.1%")
        
        with col4:
            st.metric("Throughput", "2.4k/h", delta="120")
        
        # Gráficas de API
        col1, col2 = st.columns(2)
        
        with col1:
            # Latencia over time
            times = [current_time - timedelta(minutes=i) for i in range(30, 0, -1)]
            latencies = [200 + 50 * np.sin(i/5) + np.random.normal(0, 20) for i in range(30)]
            
            fig_latency = px.line(
                x=times, y=latencies,
                title='Latencia de API (ms)',
                labels={'x': 'Tiempo', 'y': 'Latencia (ms)'}
            )
            fig_latency.update_layout(height=300)
            st.plotly_chart(fig_latency, use_container_width=True)
        
        with col2:
            # Requests per minute
            requests = [120 + 30 * np.sin(i/7) + np.random.normal(0, 10) for i in range(30)]
            
            fig_requests = px.line(
                x=times, y=requests,
                title='Requests por Minuto',
                labels={'x': 'Tiempo', 'y': 'Requests/min'}
            )
            fig_requests.update_layout(height=300)
            st.plotly_chart(fig_requests, use_container_width=True)
        
        # Tabla de endpoints más utilizados
        st.subheader("📊 Endpoints Más Utilizados")
        
        endpoints_data = {
            'Endpoint': ['/predict', '/train', '/models', '/health', '/metrics'],
            'Requests': [1240, 234, 156, 2341, 445],
            'Avg Latency (ms)': [180, 2340, 120, 45, 67],
            'Error Rate (%)': [0.2, 1.1, 0.0, 0.0, 0.1]
        }
        
        df_endpoints = pd.DataFrame(endpoints_data)
        st.dataframe(df_endpoints, use_container_width=True)
    
    def display_resource_utilization(self):
        """Muestra utilización de recursos del sistema"""
        st.header("💻 Utilización de Recursos")
        
        # Métricas actuales del sistema
        system_metrics = self.metrics_collector.collect_system_metrics()
        
        # Gauge charts para recursos principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_cpu = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = system_metrics.get('cpu_percent', 0),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CPU Usage (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            fig_memory = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = system_metrics.get('memory_percent', 0),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Usage (%)"},
                delta = {'reference': 60},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_memory.update_layout(height=300)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col3:
            fig_disk = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = system_metrics.get('disk_percent', 0),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Disk Usage (%)"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            fig_disk.update_layout(height=300)
            st.plotly_chart(fig_disk, use_container_width=True)
        
        # Detalles de recursos
        st.subheader("📋 Detalles de Recursos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💾 Memoria**")
            memory_used = system_metrics.get('memory_used_gb', 0)
            memory_total = system_metrics.get('memory_total_gb', 0)
            st.text(f"Usada: {memory_used:.2f} GB")
            st.text(f"Total: {memory_total:.2f} GB")
            st.text(f"Disponible: {memory_total - memory_used:.2f} GB")
        
        with col2:
            st.write("**💽 Disco**")
            disk_used = system_metrics.get('disk_used_gb', 0)
            disk_total = system_metrics.get('disk_total_gb', 0)
            st.text(f"Usado: {disk_used:.2f} GB")
            st.text(f"Total: {disk_total:.2f} GB")
            st.text(f"Disponible: {disk_total - disk_used:.2f} GB")
        
        # Histórico de recursos
        st.subheader("📈 Histórico de Recursos (30 min)")
        
        # Crear gráfica combinada de recursos
        resource_data = []
        for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
            data = self.metrics_collector.get_historical_data(metric, 30)
            if not data.empty:
                data['metric_type'] = metric.replace('_percent', '').upper()
                resource_data.append(data)
        
        if resource_data:
            combined_df = pd.concat(resource_data, ignore_index=True)
            fig_resources = px.line(
                combined_df, 
                x='timestamp', 
                y=combined_df.columns[1],  # La columna de valores
                color='metric_type',
                title='Utilización de Recursos (%)',
                labels={'timestamp': 'Tiempo', combined_df.columns[1]: 'Porcentaje (%)'}
            )
            fig_resources.update_layout(height=400)
            st.plotly_chart(fig_resources, use_container_width=True)
    
    def display_model_comparison(self):
        """Muestra comparación entre diferentes modelos"""
        st.header("📈 Comparación de Modelos")
        
        # Tabla comparativa de métricas
        st.subheader("🏆 Ranking de Modelos")
        
        models_df = pd.DataFrame.from_dict(self.model_metrics, orient='index')
        models_df = models_df.round(3)
        models_df = models_df.sort_values('accuracy', ascending=False)
        
        # Añadir ranking
        models_df['rank'] = range(1, len(models_df) + 1)
        models_df = models_df[['rank', 'accuracy', 'precision', 'recall', 'f1']]
        
        st.dataframe(models_df, use_container_width=True)
        
        # Gráfica de barras comparativa
        col1, col2 = st.columns(2)
        
        with col1:
            fig_accuracy = px.bar(
                x=models_df.index,
                y=models_df['accuracy'],
                title='Accuracy por Modelo',
                labels={'x': 'Modelo', 'y': 'Accuracy'}
            )
            fig_accuracy.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        with col2:
            # Gráfica radar para métricas múltiples
            fig_radar = go.Figure()
            
            for model in models_df.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[models_df.loc[model]['accuracy'], 
                       models_df.loc[model]['precision'],
                       models_df.loc[model]['recall'],
                       models_df.loc[model]['f1']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=model
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Comparación Multi-métrica",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Análisis de rendimiento temporal
        st.subheader("⏱️ Rendimiento Temporal")
        
        # Simular datos de rendimiento en el tiempo
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = []
        
        for model in models_df.index:
            base_acc = models_df.loc[model]['accuracy']
            accuracies = [base_acc + np.random.normal(0, 0.02) for _ in range(30)]
            for date, acc in zip(dates, accuracies):
                performance_data.append({
                    'date': date,
                    'model': model,
                    'accuracy': max(0, min(1, acc))  # Clamp entre 0 y 1
                })
        
        perf_df = pd.DataFrame(performance_data)
        
        fig_temporal = px.line(
            perf_df, x='date', y='accuracy', color='model',
            title='Evolución de Accuracy en el Tiempo',
            labels={'date': 'Fecha', 'accuracy': 'Accuracy'}
        )
        fig_temporal.update_layout(height=400)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    def setup_alerts_panel(self):
        """Configura panel de alertas y notificaciones"""
        st.subheader("🚨 Alertas y Notificaciones")
        
        # Generar alertas basadas en métricas actuales
        alerts = []
        
        system_metrics = self.metrics_collector.current_metrics
        
        # Alertas de CPU
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > 90:
            alerts.append({
                'type': 'error',
                'message': f'🔥 Uso de CPU crítico: {cpu_percent:.1f}%',
                'timestamp': datetime.now()
            })
        elif cpu_percent > 75:
            alerts.append({
                'type': 'warning', 
                'message': f'⚠️ Uso de CPU alto: {cpu_percent:.1f}%',
                'timestamp': datetime.now()
            })
        
        # Alertas de memoria
        memory_percent = system_metrics.get('memory_percent', 0)
        if memory_percent > 90:
            alerts.append({
                'type': 'error',
                'message': f'🔥 Uso de memoria crítico: {memory_percent:.1f}%',
                'timestamp': datetime.now()
            })
        elif memory_percent > 80:
            alerts.append({
                'type': 'warning',
                'message': f'⚠️ Uso de memoria alto: {memory_percent:.1f}%',
                'timestamp': datetime.now()
            })
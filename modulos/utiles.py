import streamlit as st
import ray
import json
import time
import logging
import os
from datetime import datetime, timedelta
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ray_utils')

def get_unique_key(base_key):
    """Genera una key única incrementando un contador"""
    counter = 0
    unique_key = f'{base_key}_{counter}'
    while unique_key in st.session_state:
        counter += 1
        unique_key = f'{base_key}_{counter}'
    return unique_key

def initialize_session_state():
    """Inicializa todas las variables de session_state necesarias"""
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'current_leader' not in st.session_state:
        st.session_state.current_leader = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

def load_custom_styles():
    """Carga los estilos CSS personalizados para la aplicación"""
    st.markdown('<style>/* Custom styles here */</style>', unsafe_allow_html=True)

def detect_current_ray_leader():
    """Detecta quién es el líder actual del clúster Ray"""
    try:
        leader_info = ray.cluster_resources().get('leader', None)
        if leader_info:
            st.session_state.current_leader = leader_info
            logger.info(f'Current Ray leader: {leader_info}')
        else:
            logger.warning('No leader detected.')
    except Exception as e:
        logger.error(f'Error detecting leader: {e}')

def connect_to_ray_cluster(max_retries=3):
    """Conecta al clúster Ray con manejo de redirección al nuevo líder"""
    for attempt in range(max_retries):
        try:
            ray.init(address='auto')
            logger.info('Connected to Ray cluster successfully.')
            return
        except Exception as e:
            logger.warning(f'Attempt {attempt + 1} failed: {e}')
            time.sleep(2)
    logger.error('Failed to connect to Ray cluster after multiple attempts.')

def save_system_metrics_history(metrics):
    """Guarda las métricas del sistema en un archivo de historial"""
    file_path = 'system_metrics_history.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    history.append(metrics)
    with open(file_path, 'w') as f:
        json.dump(history, f)
    logger.info('System metrics saved successfully.')

def load_system_metrics_history():
    """Carga el historial de métricas del sistema"""
    file_path = 'system_metrics_history.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    logger.warning('No metrics history found.')
    return []

def get_metrics_for_timeframe(hours=12):
    """Obtiene métricas para un marco de tiempo específico"""
    history = load_system_metrics_history()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    return [metric for metric in history if datetime.fromisoformat(metric['timestamp']) > cutoff_time]

def start_metrics_collection():
    """Inicia la recolección de métricas del sistema"""
    while True:
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
        }
        save_system_metrics_history(metrics)
        time.sleep(60)  # Recolectar cada minuto
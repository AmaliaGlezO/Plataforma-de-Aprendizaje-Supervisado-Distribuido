"""
Plataforma de Aprendizaje Supervisado Distribuido.
Sistema completo para training y serving de modelos ML en Ray.
"""

# Importaciones principales
from .ray_cluster import ClusterManager
from .ml_engine import TrainingOrchestrator, ModelServer
from .api import create_app
from .monitoring import MonitoringDashboard
from .utils import setup_logger, ConfigLoader

# Información del proyecto
__title__ = "Distributed ML Platform"
__version__ = "1.0.0"
__author__ = "Tu Nombre"
__description__ = "Plataforma distribuida para ML supervisado con Ray"

# Configuración global
PLATFORM_CONFIG = {
    "ray": {
        "head_port": 10001,
        "dashboard_port": 8265
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000
    },
    "monitoring": {
        "port": 8050
    }
}

__all__ = [
    "ClusterManager",
    "TrainingOrchestrator",
    "ModelServer",
    "create_app",
    "MonitoringDashboard",
    "setup_logger",
    "ConfigLoader",
    "PLATFORM_CONFIG"
]
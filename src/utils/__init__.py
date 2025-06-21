"""
Utilidades compartidas del sistema.
Funciones helper para logging, storage, health checks y configuración.
"""

from .storage import StorageManager, ModelStorage
from .logger import setup_logger, get_logger
from .health_checker import HealthChecker, check_system_health
from .config_loader import ConfigLoader, load_config

# Configuración de utilidades
UTILS_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "storage_backend": "filesystem",  # o "s3", "gcs"
    "health_check_interval": 30  # segundos
}

__all__ = [
    "StorageManager",
    "ModelStorage",
    "setup_logger",
    "get_logger",
    "HealthChecker", 
    "check_system_health",
    "ConfigLoader",
    "load_config",
    "UTILS_CONFIG"
]
"""
API REST para la plataforma de ML distribuido.
Proporciona endpoints para training, predicción y monitoreo.
"""

from .main import create_app

# Configuración por defecto de la API
DEFAULT_API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "debug": False,
    "reload": False,
    "cors_origins": ["*"],
    "rate_limit": "100/minute"
}

__all__ = [
    "create_app",
    "DEFAULT_API_CONFIG"
]
"""
Rutas de la API REST.
Organiza endpoints por funcionalidad: training, predicción, modelos, monitoreo.
"""

from .training import router as training_router
from .prediction import router as prediction_router  
from .models import router as models_router
from .monitoring import router as monitoring_router

# Lista de todos los routers para registro automático
ROUTERS = [
    (training_router, "/training", "training"),
    (prediction_router, "/predict", "prediction"),
    (models_router, "/models", "models"),
    (monitoring_router, "/monitoring", "monitoring")
]

__all__ = [
    "training_router",
    "prediction_router", 
    "models_router",
    "monitoring_router",
    "ROUTERS"
]
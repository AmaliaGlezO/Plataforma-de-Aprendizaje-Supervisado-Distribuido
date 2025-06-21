"""
Motor de Machine Learning distribuido.
Maneja carga de datos, entrenamiento, validación y serving de modelos.
"""

from .data_loader import DataLoader, DistributedDataLoader
from .model_factory import ModelFactory, SupportedModels
from .training_orchestrator import TrainingOrchestrator
from .model_validator import ModelValidator, ValidationMetrics
from .model_server import ModelServer

# Tipos de modelos soportados
SUPPORTED_ALGORITHMS = [
    "RandomForestClassifier",
    "LogisticRegression", 
    "SVC",
    "GradientBoostingClassifier",
    "DecisionTreeClassifier",
    "KNeighborsClassifier"
]

# Métricas de evaluación disponibles
AVAILABLE_METRICS = [
    "accuracy",
    "precision",
    "recall", 
    "f1",
    "roc_auc",
    "confusion_matrix"
]

__all__ = [
    "DataLoader",
    "DistributedDataLoader",
    "ModelFactory",
    "SupportedModels",
    "TrainingOrchestrator",
    "ModelValidator",
    "ValidationMetrics",
    "ModelServer",
    "SUPPORTED_ALGORITHMS",
    "AVAILABLE_METRICS"
]
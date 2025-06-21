"""
Esquemas Pydantic para operaciones de entrenamiento de modelos ML.
Define la estructura de datos para requests y responses del training distribuido.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator


class TrainingStatus(str, Enum):
    """Estados posibles del entrenamiento."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(str, Enum):
    """Tipos de modelos ML soportados."""
    RANDOM_FOREST = "RandomForestClassifier"
    LOGISTIC_REGRESSION = "LogisticRegression"
    SVM = "SVC"
    GRADIENT_BOOSTING = "GradientBoostingClassifier"
    DECISION_TREE = "DecisionTreeClassifier"
    KNN = "KNeighborsClassifier"


class DataSplitConfig(BaseModel):
    """Configuración para división de datos."""
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Proporción para test set")
    validation_size: float = Field(0.1, ge=0.0, le=0.3, description="Proporción para validation set")
    random_state: Optional[int] = Field(42, description="Semilla para reproducibilidad")
    stratify: bool = Field(True, description="Estratificar la división por clases")
    
    @validator('test_size', 'validation_size')
    def validate_proportions(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Las proporciones deben estar entre 0 y 1")
        return v
    
    @root_validator
    def validate_total_split(cls, values):
        test_size = values.get('test_size', 0.2)
        validation_size = values.get('validation_size', 0.1)
        if test_size + validation_size >= 1.0:
            raise ValueError("La suma de test_size y validation_size debe ser menor a 1.0")
        return values


class ModelConfig(BaseModel):
    """Configuración específica de cada modelo ML."""
    model_type: ModelType = Field(..., description="Tipo de modelo a entrenar")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hiperparámetros del modelo")
    cross_validation_folds: int = Field(5, ge=2, le=20, description="Número de folds para CV")
    scoring_metric: str = Field("accuracy", description="Métrica principal de evaluación")
    
    @validator('hyperparameters')
    def validate_hyperparameters(cls, v, values):
        model_type = values.get('model_type')
        if model_type and v:
            # Validaciones específicas por tipo de modelo
            if model_type == ModelType.RANDOM_FOREST:
                allowed_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'random_state']
            elif model_type == ModelType.LOGISTIC_REGRESSION:
                allowed_params = ['C', 'penalty', 'solver', 'max_iter', 'random_state']
            elif model_type == ModelType.SVM:
                allowed_params = ['C', 'kernel', 'gamma', 'degree', 'random_state']
            elif model_type == ModelType.GRADIENT_BOOSTING:
                allowed_params = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'random_state']
            elif model_type == ModelType.DECISION_TREE:
                allowed_params = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion', 'random_state']
            elif model_type == ModelType.KNN:
                allowed_params = ['n_neighbors', 'weights', 'algorithm', 'p', 'metric']
            else:
                allowed_params = []
            
            # Validar que solo se usen parámetros permitidos
            invalid_params = set(v.keys()) - set(allowed_params)
            if invalid_params:
                raise ValueError(f"Parámetros no válidos para {model_type}: {invalid_params}")
        
        return v


class DistributedConfig(BaseModel):
    """Configuración para entrenamiento distribuido."""
    num_workers: int = Field(2, ge=1, le=50, description="Número de workers Ray")
    resources_per_worker: Dict[str, float] = Field(
        default_factory=lambda: {"cpu": 1.0, "memory": 2.0},
        description="Recursos por worker"
    )
    max_retries: int = Field(3, ge=1, le=10, description="Reintentos máximos por tarea")
    timeout_minutes: int = Field(60, ge=5, le=600, description="Timeout por entrenamiento")
    parallel_trials: int = Field(4, ge=1, le=20, description="Entrenamientos paralelos")


class TrainingRequest(BaseModel):
    """Request para iniciar entrenamiento de modelos."""
    dataset_path: str = Field(..., description="Ruta al dataset")
    target_column: str = Field(..., description="Nombre de la columna objetivo")
    feature_columns: Optional[List[str]] = Field(None, description="Columnas a usar como features")
    models_config: List[ModelConfig] = Field(..., min_items=1, description="Configuración de modelos a entrenar")
    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig, description="Configuración de división de datos")
    distributed_config: DistributedConfig = Field(default_factory=DistributedConfig, description="Configuración distribuida")
    experiment_name: str = Field(..., min_length=1, max_length=100, description="Nombre del experimento")
    description: Optional[str] = Field(None, max_length=500, description="Descripción del experimento")
    save_model: bool = Field(True, description="Guardar modelos entrenados")
    
    @validator('dataset_path')
    def validate_dataset_path(cls, v):
        if not v.endswith(('.csv', '.parquet', '.json')):
            raise ValueError("Dataset debe ser .csv, .parquet o .json")
        return v
    
    @validator('experiment_name')
    def validate_experiment_name(cls, v):
        # Validar que sea un nombre válido para archivos
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Nombre de experimento solo puede contener letras, números, _ y -")
        return v


class ModelMetrics(BaseModel):
    """Métricas de evaluación de un modelo."""
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    roc_auc: Optional[float] = Field(None, ge=0.0, le=1.0)
    confusion_matrix: List[List[int]] = Field(..., description="Matriz de confusión")
    classification_report: Dict[str, Any] = Field(..., description="Reporte detallado de clasificación")
    cross_val_scores: List[float] = Field(..., description="Scores de validación cruzada")
    training_time_seconds: float = Field(..., ge=0.0, description="Tiempo de entrenamiento")


class TrainedModel(BaseModel):
    """Información de un modelo entrenado."""
    model_id: str = Field(..., description="ID único del modelo")
    model_type: ModelType = Field(..., description="Tipo de modelo")
    hyperparameters: Dict[str, Any] = Field(..., description="Hiperparámetros usados")
    metrics: ModelMetrics = Field(..., description="Métricas de evaluación")
    feature_columns: List[str] = Field(..., description="Columnas usadas como features")
    target_column: str = Field(..., description="Columna objetivo")
    model_path: Optional[str] = Field(None, description="Ruta donde se guardó el modelo")
    created_at: datetime = Field(..., description="Timestamp de creación")
    worker_node: Optional[str] = Field(None, description="Nodo que entrenó el modelo")


class TrainingProgress(BaseModel):
    """Progreso del entrenamiento."""
    total_models: int = Field(..., ge=1)
    completed_models: int = Field(..., ge=0)
    failed_models: int = Field(..., ge=0)
    current_model: Optional[str] = Field(None, description="Modelo actualmente en entrenamiento")
    estimated_remaining_minutes: Optional[float] = Field(None, ge=0.0)
    
    @validator('completed_models', 'failed_models')
    def validate_progress(cls, v, values):
        total = values.get('total_models', 0)
        if 'completed_models' in values:
            completed = values['completed_models']
            failed = v if cls.__name__ == 'failed_models' else values.get('failed_models', 0)
            if completed + failed > total:
                raise ValueError("Modelos completados + fallidos no puede exceder el total")
        return v


class TrainingResponse(BaseModel):
    """Response del entrenamiento iniciado."""
    job_id: str = Field(..., description="ID único del job de entrenamiento")
    experiment_name: str = Field(..., description="Nombre del experimento")
    status: TrainingStatus = Field(..., description="Estado actual del entrenamiento")
    message: str = Field(..., description="Mensaje descriptivo")
    started_at: datetime = Field(..., description="Timestamp de inicio")
    progress: TrainingProgress = Field(..., description="Progreso del entrenamiento")
    ray_job_id: Optional[str] = Field(None, description="ID del job en Ray")


class TrainingResult(BaseModel):
    """Resultado completo del entrenamiento."""
    job_id: str = Field(..., description="ID del job")
    experiment_name: str = Field(..., description="Nombre del experimento")
    status: TrainingStatus = Field(..., description="Estado final")
    trained_models: List[TrainedModel] = Field(..., description="Modelos entrenados exitosamente")
    failed_models: List[Dict[str, Any]] = Field(default_factory=list, description="Modelos que fallaron")
    best_model: Optional[TrainedModel] = Field(None, description="Mejor modelo por métrica principal")
    total_training_time_minutes: float = Field(..., ge=0.0)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None, description="Mensaje de error si falló")
    
    @validator('best_model')
    def validate_best_model(cls, v, values):
        if v and values.get('trained_models'):
            # Verificar que el mejor modelo esté en la lista de entrenados
            trained_ids = [m.model_id for m in values['trained_models']]
            if v.model_id not in trained_ids:
                raise ValueError("El mejor modelo debe estar en la lista de modelos entrenados")
        return v


class BatchTrainingRequest(BaseModel):
    """Request para entrenamiento de múltiples datasets."""
    training_requests: List[TrainingRequest] = Field(..., min_items=1, max_items=10)
    sequential: bool = Field(False, description="Ejecutar secuencialmente o en paralelo")
    stop_on_error: bool = Field(False, description="Detener si falla algún entrenamiento")
    
    @validator('training_requests')
    def validate_unique_experiments(cls, v):
        experiment_names = [req.experiment_name for req in v]
        if len(experiment_names) != len(set(experiment_names)):
            raise ValueError("Los nombres de experimento deben ser únicos")
        return v


class BatchTrainingResponse(BaseModel):
    """Response del entrenamiento por lotes."""
    batch_id: str = Field(..., description="ID único del lote")
    individual_jobs: List[TrainingResponse] = Field(..., description="Jobs individuales")
    status: TrainingStatus = Field(..., description="Estado del lote")
    started_at: datetime = Field(...)
    sequential_mode: bool = Field(..., description="Modo secuencial o paralelo")


# Schemas para consulta de estado
class JobStatusRequest(BaseModel):
    """Request para consultar estado de job."""
    job_id: str = Field(..., description="ID del job a consultar")


class JobStatusResponse(BaseModel):
    """Response con estado actual del job."""
    job_id: str = Field(...)
    status: TrainingStatus = Field(...)
    progress: TrainingProgress = Field(...)
    result: Optional[TrainingResult] = Field(None, description="Resultado si está completado")
    last_updated: datetime = Field(...)


# Schemas para gestión de experimentos
class ExperimentListResponse(BaseModel):
    """Lista de experimentos."""
    experiments: List[Dict[str, Any]] = Field(..., description="Lista de experimentos")
    total_count: int = Field(..., ge=0)
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=100)


class DeleteExperimentRequest(BaseModel):
    """Request para eliminar experimento."""
    experiment_name: str = Field(..., description="Nombre del experimento")
    delete_models: bool = Field(False, description="Eliminar también los archivos de modelos")
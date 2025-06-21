"""
Esquemas Pydantic para operaciones de predicción/inferencia.
Define la estructura de datos para requests y responses del serving de modelos.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np


class PredictionType(str, Enum):
    """Tipos de predicción soportados."""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"


class OutputFormat(str, Enum):
    """Formatos de salida para predicciones."""
    JSON = "json"
    CSV = "csv"
    NUMPY = "numpy"


class ModelStatus(str, Enum):
    """Estados del modelo en el servidor."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


class PredictionInput(BaseModel):
    """Input de datos para predicción individual."""
    features: Dict[str, Union[float, int, str, bool]] = Field(..., description="Features del input")
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features no puede estar vacío")
        
        # Validar tipos de datos permitidos
        for key, value in v.items():
            if not isinstance(value, (int, float, str, bool)) and value is not None:
                raise ValueError(f"Tipo de dato no soportado para feature '{key}': {type(value)}")
        
        return v


class PredictionRequest(BaseModel):
    """Request para predicción individual."""
    model_id: str = Field(..., description="ID del modelo a usar")
    input_data: PredictionInput = Field(..., description="Datos de entrada")
    return_probabilities: bool = Field(False, description="Retornar probabilidades de clase")
    return_feature_importance: bool = Field(False, description="Retornar importancia de features")
    
    @validator('model_id')
    def validate_model_id(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("model_id no puede estar vacío")
        return v.strip()


class ClassProbability(BaseModel):
    """Probabilidad por clase."""
    class_name: Union[str, int] = Field(..., description="Nombre o valor de la clase")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de la clase")


class FeatureImportance(BaseModel):
    """Importancia de una feature."""
    feature_name: str = Field(..., description="Nombre de la feature")
    importance: float = Field(..., ge=0.0, description="Valor de importancia")


class PredictionOutput(BaseModel):
    """Output de una predicción."""
    predicted_class: Union[str, int, float] = Field(..., description="Clase predicha")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la predicción")
    probabilities: Optional[List[ClassProbability]] = Field(None, description="Probabilidades por clase")
    feature_importance: Optional[List[FeatureImportance]] = Field(None, description="Importancia de features")


class PredictionResponse(BaseModel):
    """Response de predicción individual."""
    prediction_id: str = Field(..., description="ID único de la predicción")
    model_id: str = Field(..., description="ID del modelo usado")
    prediction: PredictionOutput = Field(..., description="Resultado de la predicción")
    processing_time_ms: float = Field(..., ge=0.0, description="Tiempo de procesamiento en ms")
    timestamp: datetime = Field(..., description="Timestamp de la predicción")
    model_version: Optional[str] = Field(None, description="Versión del modelo")


class BatchPredictionInput(BaseModel):
    """Input para predicción por lotes."""
    data: List[Dict[str, Union[float, int, str, bool]]] = Field(..., min_items=1, max_items=10000, description="Lista de inputs")
    
    @validator('data')
    def validate_batch_data(cls, v):
        if not v:
            raise ValueError("La lista de datos no puede estar vacía")
        
        # Validar que todas las muestras tengan las mismas features
        if len(v) > 1:
            first_keys = set(v[0].keys())
            for i, sample in enumerate(v[1:], 1):
                sample_keys = set(sample.keys())
                if sample_keys != first_keys:
                    raise ValueError(f"Sample {i} tiene features diferentes: {sample_keys} vs {first_keys}")
        
        # Validar tipos de datos
        for i, sample in enumerate(v):
            for key, value in sample.items():
                if not isinstance(value, (int, float, str, bool)) and value is not None:
                    raise ValueError(f"Sample {i}, feature '{key}': tipo no soportado {type(value)}")
        
        return v


class BatchPredictionRequest(BaseModel):
    """Request para predicción por lotes."""
    model_id: str = Field(..., description="ID del modelo a usar")
    batch_data: BatchPredictionInput = Field(..., description="Datos de entrada por lotes")
    return_probabilities: bool = Field(False, description="Retornar probabilidades")
    return_feature_importance: bool = Field(False, description="Retornar importancia de features")
    output_format: OutputFormat = Field(OutputFormat.JSON, description="Formato de salida")
    async_processing: bool = Field(False, description="Procesamiento asíncrono")
    
    @validator('model_id')
    def validate_model_id(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("model_id no puede estar vacío")
        return v.strip()


class BatchPredictionResponse(BaseModel):
    """Response para predicción por lotes."""
    batch_id: str = Field(..., description="ID único del lote")
    model_id: str = Field(..., description="ID del modelo usado")
    predictions: List[PredictionOutput] = Field(..., description="Lista de predicciones")
    total_samples: int = Field(..., ge=0, description="Total de muestras procesadas")
    successful_predictions: int = Field(..., ge=0, description="Predicciones exitosas")
    failed_predictions: int = Field(..., ge=0, description="Predicciones fallidas")
    processing_time_ms: float = Field(..., ge=0.0, description="Tiempo total de procesamiento")
    timestamp: datetime = Field(..., description="Timestamp del procesamiento")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Errores encontrados")
    
    @validator('successful_predictions', 'failed_predictions')
    def validate_prediction_counts(cls, v, values):
        total = values.get('total_samples', 0)
        if 'successful_predictions' in values:
            successful = values['successful_predictions']
            failed = v if 'failed_predictions' in cls.__fields__ else values.get('failed_predictions', 0)
            if successful + failed != total:
                raise ValueError("Suma de predicciones exitosas y fallidas debe igualar total")
        return v


class AsyncBatchResponse(BaseModel):
    """Response para procesamiento asíncrono por lotes."""
    job_id: str = Field(..., description="ID del job asíncrono")
    status: str = Field(..., description="Estado del procesamiento")
    estimated_completion_time: Optional[datetime] = Field(None, description="Tiempo estimado de finalización")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Porcentaje de progreso")


class StreamingPredictionRequest(BaseModel):
    """Request para predicción en streaming."""
    model_id: str = Field(..., description="ID del modelo a usar")
    stream_config: Dict[str, Any] = Field(..., description="Configuración del stream")
    buffer_size: int = Field(100, ge=1, le=10000, description="Tamaño del buffer")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Timeout por batch")


class ModelLoadRequest(BaseModel):
    """Request para cargar un modelo en memoria."""
    model_id: str = Field(..., description="ID del modelo a cargar")
    model_path: Optional[str] = Field(None, description="Path específico del modelo")
    cache_size_mb: int = Field(512, ge=64, le=8192, description="Tamaño de caché en MB")
    
    @validator('model_id')
    def validate_model_id(cls, v):
        if len(v.strip()) == 0:
            raise ValueError("model_id no puede estar vacío")
        return v.strip()


class ModelInfo(BaseModel):
    """Información de un modelo cargado."""
    model_id: str = Field(..., description="ID del modelo")
    model_type: str = Field(..., description="Tipo de modelo")
    status: ModelStatus = Field(..., description="Estado del modelo")
    feature_names: List[str] = Field(..., description="Nombres de features esperadas")
    target_classes: List[Union[str, int]] = Field(..., description="Clases objetivo")
    loaded_at: datetime = Field(..., description="Timestamp de carga")
    memory_usage_mb: float = Field(..., ge=0.0, description="Uso de memoria en MB")
    last_prediction_at: Optional[datetime] = Field(None, description="Última predicción")
    total_predictions: int = Field(0, ge=0, description="Total de predicciones realizadas")
    average_latency_ms: float = Field(0.0, ge=0.0, description="Latencia promedio")


class ModelLoadResponse(BaseModel):
    """Response de carga de modelo."""
    model_id: str = Field(..., description="ID del modelo")
    status: ModelStatus = Field(..., description="Estado de la carga")
    message: str = Field(..., description="Mensaje descriptivo")
    model_info: Optional[ModelInfo] = Field(None, description="Información del modelo si se cargó exitosamente")
    loading_time_ms: float = Field(..., ge=0.0, description="Tiempo de carga en ms")


class ModelListResponse(BaseModel):
    """Response con lista de modelos disponibles."""
    loaded_models: List[ModelInfo] = Field(..., description="Modelos cargados en memoria")
    available_models: List[Dict[str, Any]] = Field(..., description="Modelos disponibles para cargar")
    total_memory_usage_mb: float = Field(..., ge=0.0, description="Uso total de memoria")
    max_memory_limit_mb: float = Field(..., ge=0.0, description="Límite máximo de memoria")


class ModelUnloadRequest(BaseModel):
    """Request para descargar modelo de memoria."""
    model_id: str = Field(..., description="ID del modelo a descargar")
    force: bool = Field(False, description="Forzar descarga aunque esté en uso")


class ModelUnloadResponse(BaseModel):
    """Response de descarga de modelo."""
    model_id: str = Field(..., description="ID del modelo")
    status: str = Field(..., description="Estado de la descarga")
    message: str = Field(..., description="Mensaje descriptivo")
    memory_freed_mb: float = Field(..., ge=0.0, description="Memoria liberada en MB")


class PredictionMetrics(BaseModel):
    """Métricas de predicción en tiempo real."""
    model_id: str = Field(..., description="ID del modelo")
    total_predictions: int = Field(..., ge=0, description="Total de predicciones")
    predictions_per_second: float = Field(..., ge=0.0, description="Predicciones por segundo")
    average_latency_ms: float = Field(..., ge=0.0, description="Latencia promedio")
    p95_latency_ms: float = Field(..., ge=0.0, description="Latencia percentil 95")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Tasa de errores")
    memory_usage_mb: float = Field(..., ge=0.0, description="Uso de memoria")
    last_updated: datetime = Field(..., description="Última actualización")


class HealthCheckResponse(BaseModel):
    """Response del health check del servicio de predicción."""
    status: str = Field(..., description="Estado general del servicio")
    models_loaded: int = Field(..., ge=0, description="Número de modelos cargados")
    total_memory_usage_mb: float = Field(..., ge=0.0, description="Uso total de memoria")
    uptime_seconds: float = Field(..., ge=0.0, description="Tiempo de actividad")
    last_prediction_time: Optional[datetime] = Field(None, description="Última predicción realizada")
    ray_cluster_status: str = Field(..., description="Estado del cluster Ray")


# Schemas para manejo de errores
class PredictionError(BaseModel):
    """Error en predicción."""
    error_code: str = Field(..., description="Código de error")
    error_message: str = Field(..., description="Mensaje de error")
    sample_index: Optional[int] = Field(None, description="Índice de la muestra que causó error")
    feature_name: Optional[str] = Field(None, description="Nombre de feature problemática")


class ValidationError(BaseModel):
    """Error de validación de datos."""
    field_name: str = Field(..., description="Campo que causó el error")
    error_message: str = Field(..., description="Mensaje de error")
    provided_value: Any = Field(..., description="Valor proporcionado")
    expected_type: str = Field(..., description="Tipo esperado")


# Schemas para configuración de serving
class ServingConfig(BaseModel):
    """Configuración del servicio de predicción."""
    max_concurrent_requests: int = Field(100, ge=1, le=10000, description="Requests concurrentes máximos")
    model_cache_size_mb: int = Field(2048, ge=256, le=16384, description="Tamaño de caché de modelos")
    prediction_timeout_seconds: int = Field(30, ge=1, le=300, description="Timeout por predicción")
    auto_unload_minutes: int = Field(60, ge=5, le=1440, description="Auto-descarga de modelos inactivos")
    enable_metrics: bool = Field(True, description="Habilitar recolección de métricas")
    log_predictions: bool = Field(False, description="Registrar todas las predicciones")


class ModelServingStats(BaseModel):
    """Estadísticas de serving por modelo."""
    model_id: str = Field(..., description="ID del modelo")
    requests_count: int = Field(..., ge=0, description="Número total de requests")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Tasa de éxito")
    avg_response_time_ms: float = Field(..., ge=0.0, description="Tiempo de respuesta promedio")
    throughput_per_minute: float = Field(..., ge=0.0, description="Throughput por minuto")
    memory_usage_mb: float = Field(..., ge=0.0, description="Uso de memoria")
    last_request_time: Optional[datetime] = Field(None, description="Última request")
    error_count: int = Field(..., ge=0, description="Número de errores")
    most_common_errors: List[str] = Field(default_factory=list, description="Errores más comunes")
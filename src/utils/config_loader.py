"""
Sistema de carga y gestión de configuraciones para la plataforma ML distribuida.
Soporta múltiples formatos (YAML, JSON, ENV) con validación y valores por defecto.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError
import configparser
from datetime import datetime


# Configuración del logger
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Excepción personalizada para errores de configuración."""
    pass


@dataclass
class ConfigSource:
    """Información sobre la fuente de una configuración."""
    file_path: Optional[str] = None
    format_type: Optional[str] = None
    loaded_at: Optional[datetime] = None
    is_default: bool = False
    env_overrides: List[str] = field(default_factory=list)


class RayClusterConfig(BaseModel):
    """Configuración del cluster Ray."""
    head_port: int = 10001
    dashboard_port: int = 8265
    redis_port: int = 6379
    worker_ports: List[int] = [10002, 10003, 10004, 10005]
    redis_password: Optional[str] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    memory_limit_gb: Optional[float] = None
    object_store_memory_gb: Optional[float] = None
    dashboard_host: str = "0.0.0.0"
    temp_dir: Optional[str] = None
    log_level: str = "INFO"
    
    class Config:
        extra = "forbid"


class APIConfig(BaseModel):
    """Configuración de la API REST."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False
    reload: bool = False
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    rate_limit: str = "100/minute"
    rate_limit_storage: str = "memory"
    timeout_seconds: int = 300
    max_request_size_mb: int = 100
    
    class Config:
        extra = "forbid"


class MonitoringConfig(BaseModel):
    """Configuración del sistema de monitoreo."""
    dashboard_port: int = 8050
    dashboard_host: str = "0.0.0.0"
    update_interval_seconds: int = 5
    retention_days: int = 30
    metrics_endpoint: str = "/metrics"
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    alert_thresholds: Dict[str, float] = {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "error_rate": 5.0
    }
    
    class Config:
        extra = "forbid"


class MLEngineConfig(BaseModel):
    """Configuración del motor de ML."""
    default_test_size: float = 0.2
    default_validation_size: float = 0.1
    default_cv_folds: int = 5
    max_training_time_minutes: int = 120
    auto_save_models: bool = True
    model_storage_path: str = "./data/models"
    checkpoint_interval_minutes: int = 10
    supported_formats: List[str] = ["csv", "parquet", "json"]
    max_dataset_size_gb: float = 10.0
    
    class Config:
        extra = "forbid"


class StorageConfig(BaseModel):
    """Configuración de almacenamiento."""
    backend: str = "filesystem"  # filesystem, s3, gcs
    base_path: str = "./data"
    models_path: str = "./data/models"
    datasets_path: str = "./data/datasets"
    checkpoints_path: str = "./data/checkpoints"
    logs_path: str = "./logs"
    
    # Configuración S3 (opcional)
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    # Configuración GCS (opcional)
    gcs_bucket: Optional[str] = None
    gcs_credentials_path: Optional[str] = None
    
    class Config:
        extra = "forbid"


class LoggingConfig(BaseModel):
    """Configuración del sistema de logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler: bool = True
    console_handler: bool = True
    log_file: str = "./logs/platform.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    json_format: bool = False
    
    class Config:
        extra = "forbid"


class SecurityConfig(BaseModel):
    """Configuración de seguridad."""
    enable_auth: bool = False
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    token_expire_minutes: int = 30
    enable_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    rate_limiting: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    class Config:
        extra = "forbid"


@dataclass
class PlatformConfig:
    """Configuración completa de la plataforma."""
    ray_cluster: RayClusterConfig = field(default_factory=RayClusterConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ml_engine: MLEngineConfig = field(default_factory=MLEngineConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Metadatos
    _source: ConfigSource = field(default_factory=ConfigSource)
    _environment: str = "development"


class ConfigLoader:
    """
    Cargador principal de configuraciones con soporte para múltiples fuentes
    y formatos, incluyendo variables de entorno y valores por defecto.
    """
    
    def __init__(self, config_dir: str = "./config"):
        """
        Inicializa el cargador de configuraciones.
        
        Args:
            config_dir: Directorio base donde buscar archivos de configuración
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config_cache: Dict[str, Any] = {}
        self._env_prefix = "PLATFORM_"
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, 
                   config_name: Optional[str] = None,
                   environment: str = "development") -> PlatformConfig:
        """
        Carga la configuración completa de la plataforma.
        
        Args:
            config_name: Nombre específico del archivo de configuración
            environment: Entorno (development, staging, production)
            
        Returns:
            PlatformConfig: Configuración completa validada
        """
        try:
            # Determinar archivo de configuración
            if config_name:
                config_files = [f"{config_name}.yaml", f"{config_name}.yml", f"{config_name}.json"]
            else:
                config_files = [
                    f"config.{environment}.yaml",
                    f"config.{environment}.yml", 
                    f"config.{environment}.json",
                    "config.yaml",
                    "config.yml",
                    "config.json"
                ]
            
            # Intentar cargar archivo de configuración
            config_data = {}
            source = ConfigSource(is_default=True)
            
            for config_file in config_files:
                config_path = self.config_dir / config_file
                if config_path.exists():
                    config_data = self._load_config_file(config_path)
                    source = ConfigSource(
                        file_path=str(config_path),
                        format_type=config_path.suffix[1:],
                        loaded_at=datetime.now(),
                        is_default=False
                    )
                    self.logger.info(f"Configuración cargada desde: {config_path}")
                    break
            else:
                self.logger.warning("No se encontró archivo de configuración, usando valores por defecto")
            
            # Aplicar overrides de variables de entorno
            env_overrides = self._load_env_overrides()
            if env_overrides:
                config_data = self._merge_configs(config_data, env_overrides)
                source.env_overrides = list(env_overrides.keys())
                self.logger.info(f"Aplicados {len(env_overrides)} overrides de variables de entorno")
            
            # Crear configuración tipada
            platform_config = self._create_platform_config(config_data, environment)
            platform_config._source = source
            platform_config._environment = environment
            
            # Validar configuración
            self._validate_config(platform_config)
            
            # Cachear configuración
            self._config_cache[environment] = platform_config
            
            self.logger.info(f"Configuración cargada exitosamente para entorno: {environment}")
            return platform_config
            
        except Exception as e:
            self.logger.error(f"Error cargando configuración: {e}")
            raise ConfigurationError(f"No se pudo cargar la configuración: {e}")
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Carga un archivo de configuración específico."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Formato de archivo no soportado: {config_path.suffix}")
                    
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parseando YAML en {config_path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Error parseando JSON en {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error leyendo archivo {config_path}: {e}")
    
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Carga overrides desde variables de entorno."""
        overrides = {}
        
        # Mapeo de variables de entorno a rutas de configuración
        env_mappings = {
            # Ray Cluster
            f"{self._env_prefix}RAY_HEAD_PORT": "ray_cluster.head_port",
            f"{self._env_prefix}RAY_DASHBOARD_PORT": "ray_cluster.dashboard_port",
            f"{self._env_prefix}RAY_REDIS_PASSWORD": "ray_cluster.redis_password",
            f"{self._env_prefix}RAY_NUM_CPUS": "ray_cluster.num_cpus",
            f"{self._env_prefix}RAY_NUM_GPUS": "ray_cluster.num_gpus",
            
            # API
            f"{self._env_prefix}API_HOST": "api.host",
            f"{self._env_prefix}API_PORT": "api.port",
            f"{self._env_prefix}API_WORKERS": "api.workers",
            f"{self._env_prefix}API_DEBUG": "api.debug",
            
            # Monitoring
            f"{self._env_prefix}MONITORING_PORT": "monitoring.dashboard_port",
            f"{self._env_prefix}MONITORING_UPDATE_INTERVAL": "monitoring.update_interval_seconds",
            
            # Storage
            f"{self._env_prefix}STORAGE_BACKEND": "storage.backend",
            f"{self._env_prefix}STORAGE_BASE_PATH": "storage.base_path",
            f"{self._env_prefix}S3_BUCKET": "storage.s3_bucket",
            f"{self._env_prefix}S3_REGION": "storage.s3_region",
            
            # Security
            f"{self._env_prefix}SECRET_KEY": "security.secret_key",
            f"{self._env_prefix}ENABLE_AUTH": "security.enable_auth",
            f"{self._env_prefix}ENABLE_HTTPS": "security.enable_https",
            
            # Logging
            f"{self._env_prefix}LOG_LEVEL": "logging.level",
            f"{self._env_prefix}LOG_FILE": "logging.log_file",
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convertir tipos de datos
                converted_value = self._convert_env_value(env_value, config_path)
                self._set_nested_value(overrides, config_path, converted_value)
        
        return overrides
    
    def _convert_env_value(self, value: str, config_path: str) -> Any:
        """Convierte valores de variables de entorno a tipos apropiados."""
        # Valores booleanos
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Valores numéricos
        if config_path.endswith(('port', 'workers', 'cpus', 'gpus', 'interval', 'minutes')):
            try:
                return int(value)
            except ValueError:
                pass
        
        if config_path.endswith(('size', 'limit', 'threshold')):
            try:
                return float(value)
            except ValueError:
                pass
        
        # Lista separada por comas
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Establece un valor en una estructura anidada usando notación de puntos."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge recursivo de configuraciones."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_platform_config(self, config_data: Dict[str, Any], environment: str) -> PlatformConfig:
        """Crea la configuración de plataforma tipada."""
        try:
            # Extraer secciones de configuración
            ray_config_data = config_data.get('ray_cluster', {})
            api_config_data = config_data.get('api', {})
            monitoring_config_data = config_data.get('monitoring', {})
            ml_engine_config_data = config_data.get('ml_engine', {})
            storage_config_data = config_data.get('storage', {})
            logging_config_data = config_data.get('logging', {})
            security_config_data = config_data.get('security', {})
            
            # Crear configuraciones tipadas
            return PlatformConfig(
                ray_cluster=RayClusterConfig(**ray_config_data),
                api=APIConfig(**api_config_data),
                monitoring=MonitoringConfig(**monitoring_config_data),
                ml_engine=MLEngineConfig(**ml_engine_config_data),
                storage=StorageConfig(**storage_config_data),
                logging=LoggingConfig(**logging_config_data),
                security=SecurityConfig(**security_config_data)
            )
            
        except ValidationError as e:
            raise ConfigurationError(f"Error de validación en configuración: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error creando configuración tipada: {e}")
    
    def _validate_config(self, config: PlatformConfig):
        """Valida la configuración cargada."""
        # Validar puertos únicos
        ports = [
            config.ray_cluster.head_port,
            config.ray_cluster.dashboard_port,
            config.ray_cluster.redis_port,
            config.api.port,
            config.monitoring.dashboard_port
        ]
        
        if config.monitoring.enable_prometheus:
            ports.append(config.monitoring.prometheus_port)
        
        if len(ports) != len(set(ports)):
            raise ConfigurationError("Puertos duplicados detectados en la configuración")
        
        # Validar rutas de almacenamiento
        storage_paths = [
            config.storage.base_path,
            config.storage.models_path,
            config.storage.datasets_path,
            config.storage.checkpoints_path
        ]
        
        for path in storage_paths:
            try:
                Path(path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"No se pudo crear directorio {path}: {e}")
        
        # Validar configuración de seguridad
        if config.security.enable_https:
            if not config.security.ssl_cert_path or not config.security.ssl_key_path:
                raise ConfigurationError("SSL habilitado pero faltan rutas de certificados")
    
    def save_config(self, config: PlatformConfig, 
                   file_path: Optional[str] = None,
                   format_type: str = "yaml") -> str:
        """
        Guarda la configuración actual en un archivo.
        
        Args:
            config: Configuración a guardar
            file_path: Ruta del archivo (opcional)
            format_type: Formato del archivo (yaml, json)
            
        Returns:
            str: Ruta del archivo guardado
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config_dir / f"config_backup_{timestamp}.{format_type}"
        else:
            file_path = Path(file_path)
        
        # Convertir a diccionario
        config_dict = {
            "ray_cluster": config.ray_cluster.dict(),
            "api": config.api.dict(),
            "monitoring": config.monitoring.dict(),
            "ml_engine": config.ml_engine.dict(),
            "storage": config.storage.dict(),
            "logging": config.logging.dict(),
            "security": config.security.dict()
        }
        
        # Guardar archivo
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type.lower() in ['yaml', 'yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format_type.lower() == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ConfigurationError(f"Formato no soportado: {format_type}")
            
            self.logger.info(f"Configuración guardada en: {file_path}")
            return str(file_path)
            
        except Exception as e:
            raise ConfigurationError(f"Error guardando configuración: {e}")
    
    def get_cached_config(self, environment: str = "development") -> Optional[PlatformConfig]:
        """Obtiene configuración desde caché."""
        return self._config_cache.get(environment)
    
    def clear_cache(self):
        """Limpia el caché de configuraciones."""
        self._config_cache.clear()
        self.logger.info("Caché de configuraciones limpiado")
    
    def reload_config(self, environment: str = "development") -> PlatformConfig:
        """Recarga la configuración desde archivos."""
        if environment in self._config_cache:
            del self._config_cache[environment]
        return self.load_config(environment=environment)


# Funciones de conveniencia
def load_config(config_dir: str = "./config", 
               environment: str = "development") -> PlatformConfig:
    """
    Función de conveniencia para cargar configuración.
    
    Args:
        config_dir: Directorio de configuraciones
        environment: Entorno a cargar
        
    Returns:
        PlatformConfig: Configuración cargada
    """
    loader = ConfigLoader(config_dir)
    return loader.load_config(environment=environment)


def get_default_config() -> PlatformConfig:
    """Obtiene configuración con valores por defecto."""
    return PlatformConfig()


def validate_config_file(file_path: str) -> bool:
    """
    Valida si un archivo de configuración es válido.
    
    Args:
        file_path: Ruta al archivo de configuración
        
    Returns:
        bool: True si es válido, False en caso contrario
    """
    try:
        loader = ConfigLoader()
        config_data = loader._load_config_file(Path(file_path))
        loader._create_platform_config(config_data, "development")
        return True
    except Exception as e:
        logger.error(f"Archivo de configuración inválido: {e}")
        return False
"""Configuración global del sistema"""

import os
from typing import Dict, List
from pydantic import BaseSettings

class Settings(BaseSettings):
    RAY_HEAD_HOST = os.getenv("RAY_HEAD_HOST", "ray-head")
    RAY_HEAD_PORT = os.getenv("RAY_HEAD_PORT", "10001")
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    DATA_DIR = "/app/data"
    MODELS_DIR = "/app/models"

    
    # === CONFIGURACIÓN RAY ===
    def get_ray_config(self) -> Dict:
        """Configuración del clúster Ray"""
        pass
    
    def get_ray_head_address(self) -> str:
        """Dirección del nodo head"""
        pass
    
    # === CONFIGURACIÓN MODELOS ===
    def get_supported_models(self) -> List[str]:
        """Lista de modelos soportados"""
        pass
    
    def get_model_configs(self) -> Dict:
        """Configuraciones específicas por modelo"""
        pass
    
    # === CONFIGURACIÓN DATOS ===
    def get_data_paths(self) -> Dict[str, str]:
        """Rutas de datos y modelos"""
        pass
    
    def get_preprocessing_config(self) -> Dict:
        """Configuración de preprocesamiento"""
        pass
    
    # === CONFIGURACIÓN API ===
    def get_api_config(self) -> Dict:
        """Configuración FastAPI"""
        pass
    
    # === CONFIGURACIÓN DASHBOARD ===
    def get_dashboard_config(self) -> Dict:
        """Configuración Streamlit"""
        pass
    
    # === CONFIGURACIÓN TOLERANCIA A FALLOS ===
    def get_fault_tolerance_config(self) -> Dict:
        """Configuración de reintentos y recuperación"""
        pass

# Instancia global
settings = Settings()
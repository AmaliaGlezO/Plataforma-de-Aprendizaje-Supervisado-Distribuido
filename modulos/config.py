# modulos/config.py
import os

# Configuración de directorios
MODELS_DIR = "models"
RESULTS_DIR = "training_results"
INFERENCE_STATS_FILE = "inference_stats.json"

# Configuración de la API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Configuración de Ray
RAY_CONFIG = {
    "enable_fault_tolerance": True
}
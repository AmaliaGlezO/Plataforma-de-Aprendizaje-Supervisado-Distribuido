"""
Sistema de monitoreo y visualización.
Proporciona dashboard, métricas y gráficas del sistema.
"""

from .dashboard import MonitoringDashboard
from .metrics_collector import MetricsCollector, SystemMetrics
from .visualizations import ChartGenerator, create_training_plots

# Configuración de monitoreo
MONITORING_CONFIG = {
    "update_interval": 5,  # segundos
    "retention_days": 30,
    "dashboard_port": 8050,
    "metrics_endpoint": "/metrics"
}

# Tipos de métricas recolectadas
METRIC_TYPES = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_io",
    "training_progress",
    "prediction_latency",
    "model_accuracy"
]

__all__ = [
    "MonitoringDashboard",
    "MetricsCollector",
    "SystemMetrics", 
    "ChartGenerator",
    "create_training_plots",
    "MONITORING_CONFIG",
    "METRIC_TYPES"
]
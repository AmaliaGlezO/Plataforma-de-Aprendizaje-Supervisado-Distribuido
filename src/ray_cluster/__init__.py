"""
Paquete de gestión del cluster Ray distribuido.
Proporciona la infraestructura para coordinar nodos head y workers.
"""

from .head_node import RayHeadNode
from .worker_node import RayWorkerNode
from .cluster_manager import ClusterManager

# Configuración por defecto del cluster
DEFAULT_CLUSTER_CONFIG = {
    "head_port": 10001,
    "dashboard_port": 8265,
    "worker_ports": [10002, 10003, 10004],
    "redis_password": None,
    "num_cpus": None,  # Auto-detect
    "num_gpus": None,  # Auto-detect
}

# API pública del módulo
__all__ = [
    "RayHeadNode",
    "RayWorkerNode", 
    "ClusterManager",
    "DEFAULT_CLUSTER_CONFIG"
]

# Versión del módulo
__version__ = "1.0.0"
"""
GESTOR DEL CLUSTER RAY
Coordina todos los aspectos del cluster distribuido.
Integra head node, workers y orquestador de entrenamientos.
"""

import ray
import time
import logging
from typing import Dict, List, Optional
from threading import Thread, Event
from dataclasses import dataclass
from .head_node import RayHeadNode
from .worker_node import RayWorkerNode

@dataclass
class ClusterConfig:
    min_workers: int = 2
    max_workers: int = 10
    resources_per_worker: Dict[str, float] = None
    heartbeat_interval: int = 5
    scaling_strategy: str = "demand"  # "demand" or "static"

class ClusterManager:
    """Gestor central del cluster Ray"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.head_node = None
        self.workers = {}
        self.scaling_event = Event()
        self._validate_config()
        
        # Iniciar con valores por defecto
        self.config.resources_per_worker = self.config.resources_per_worker or {"CPU": 2}
    
    def _setup_logging(self):
        """Configura logging centralizado"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("cluster_manager")
    
    def _validate_config(self):
        """Valida la configuración del cluster"""
        if self.config.min_workers > self.config.max_workers:
            raise ValueError("min_workers no puede ser mayor que max_workers")
    
    def start_cluster(self):
        """Inicia el cluster Ray con todos los componentes"""
        try:
            # 1. Iniciar head node
            self.head_node = RayHeadNode({
                "min_workers": self.config.min_workers,
                "max_workers": self.config.max_workers,
                "resources_per_worker": self.config.resources_per_worker
            })
            self.head_node.start_cluster()
            
            # 2. Iniciar workers iniciales
            self._scale_workers(self.config.min_workers)
            
            # 3. Iniciar servicios auxiliares
            self._start_autoscaler()
            self._start_health_monitor()
            
            self.logger.info(f"Cluster iniciado con {self.config.min_workers} workers iniciales")
            return True
            
        except Exception as e:
            self.logger.error(f"Error iniciando cluster: {str(e)}")
            raise
    
    def _scale_workers(self, target_count: int):
        """Ajusta el número de workers al objetivo especificado"""
        current_count = len(self.workers)
        
        if target_count > self.config.max_workers:
            self.logger.warning(f"No se puede escalar más allá de {self.config.max_workers} workers")
            target_count = self.config.max_workers
        
        if target_count < self.config.min_workers:
            self.logger.warning(f"No se puede reducir más allá de {self.config.min_workers} workers")
            target_count = self.config.min_workers
        
        delta = target_count - current_count
        
        if delta > 0:
            # Añadir workers
            for _ in range(delta):
                worker = RayWorkerNode(head_host="auto")
                worker_id = worker.connect_to_head()
                self.workers[worker_id] = worker
                self.logger.info(f"Worker {worker_id} añadido al cluster")
                
        elif delta < 0:
            # Remover workers (los menos activos)
            to_remove = min(abs(delta), current_count)
            for worker_id in list(self.workers.keys())[:to_remove]:
                self.workers[worker_id].shutdown()
                del self.workers[worker_id]
                self.logger.info(f"Worker {worker_id} removido del cluster")
    
    def _start_autoscaler(self):
        """Hilo para escalado automático basado en carga"""
        def autoscaler_loop():
            while not self.scaling_event.is_set():
                try:
                    if self.config.scaling_strategy == "demand":
                        self._dynamic_scaling()
                    time.sleep(self.config.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Error en autoscaler: {str(e)}")
                    time.sleep(10)
        
        Thread(target=autoscaler_loop, daemon=True).start()
    
    def _dynamic_scaling(self):
        """Lógica de escalado basado en métricas"""
        stats = self.get_cluster_stats()
        
        # Métricas para decisión de escalado
        cpu_usage = stats["cpu_usage"]
        pending_tasks = stats["pending_tasks"]
        active_workers = stats["active_workers"]
        
        # Lógica de escalado (ajustar según necesidades)
        if cpu_usage > 0.8 and active_workers < self.config.max_workers:
            self._scale_workers(active_workers + 1)
        elif cpu_usage < 0.3 and active_workers > self.config.min_workers:
            self._scale_workers(active_workers - 1)
        
        # Escalar basado en tareas pendientes
        if pending_tasks > (active_workers * 2) and active_workers < self.config.max_workers:
            self._scale_workers(min(active_workers + 1, self.config.max_workers))
    
    def _start_health_monitor(self):
        """Monitoreo continuo de salud del cluster"""
        def health_monitor_loop():
            while not self.scaling_event.is_set():
                try:
                    self._check_worker_health()
                    time.sleep(self.config.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Error en health monitor: {str(e)}")
                    time.sleep(10)
        
        Thread(target=health_monitor_loop, daemon=True).start()
    
    def _check_worker_health(self):
        """Verifica estado de los workers y toma acciones"""
        for worker_id, worker in list(self.workers.items()):
            if not worker.is_alive():
                self.logger.warning(f"Worker {worker_id} no responde - recreando...")
                del self.workers[worker_id]
                new_worker = RayWorkerNode(head_host="auto")
                new_worker.connect_to_head()
                self.workers[new_worker.worker_id] = new_worker
    
    def get_cluster_stats(self) -> Dict:
        """Obtiene estadísticas detalladas del cluster"""
        resources = ray.cluster_resources()
        used_resources = ray.available_resources()
        
        return {
            "total_workers": len(self.workers),
            "active_workers": sum(1 for w in self.workers.values() if w.is_alive()),
            "cpu_total": resources.get("CPU", 0),
            "cpu_used": resources.get("CPU", 0) - used_resources.get("CPU", 0),
            "cpu_usage": (resources.get("CPU", 0) - used_resources.get("CPU", 0)) / max(1, resources.get("CPU", 1)),
            "gpu_total": resources.get("GPU", 0),
            "memory_total": resources.get("memory", 0),
            "memory_used": resources.get("memory", 0) - used_resources.get("memory", 0),
            "pending_tasks": self.head_node.get_pending_task_count() if self.head_node else 0,
            "last_heartbeat": time.time()
        }
    
    def optimize_resources(self, task_requirements: Dict[str, float]):
        """
        Optimiza la distribución de recursos para una tarea específica
        
        Args:
            task_requirements: {"CPU": 2, "GPU": 1, "memory": 4000}
        """
        # Implementar lógica de optimización basada en:
        # 1. Requerimientos de la tarea
        # 2. Recursos disponibles
        # 3. Ubicación de datos
        pass
    
    def shutdown_cluster(self, force: bool = False):
        """Apaga el cluster de manera controlada"""
        self.scaling_event.set()
        
        # 1. Detener todos los workers
        for worker in self.workers.values():
            worker.shutdown()
        
        # 2. Detener head node
        if self.head_node:
            self.head_node.shutdown_cluster()
        
        # 3. Liberar recursos
        ray.shutdown()
        self.logger.info("Cluster apagado correctamente")
    
    def add_worker(self, worker_id: Optional[str] = None) -> str:
        """
        Añade un worker al cluster
        Returns:
            ID del worker añadido
        """
        if len(self.workers) >= self.config.max_workers:
            raise RuntimeError("Número máximo de workers alcanzado")
        
        worker = RayWorkerNode(head_host="auto", worker_id=worker_id)
        worker_id = worker.connect_to_head()
        self.workers[worker_id] = worker
        return worker_id
    
    def remove_worker(self, worker_id: str):
        """Remueve un worker del cluster"""
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} no encontrado")
        
        self.workers[worker_id].shutdown()
        del self.workers[worker_id]
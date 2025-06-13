"""
NODO COORDINADOR RAY
Este archivo maneja el nodo principal que coordina todo el cluster Ray.
Funciones: inicializar cluster, gestionar workers, distribuir tareas.
"""

import ray
import time
import logging
from typing import Dict, List, Any
from threading import Thread
from collections import defaultdict

class RayHeadNode:
    """Gestiona el nodo coordinador del cluster Ray"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el nodo head con configuración
        
        Args:
            config: {
                "head_host": "auto" o dirección IP,
                "redis_port": 6379,
                "dashboard_port": 8265,
                "min_workers": 1,
                "max_workers": 10,
                "resources_per_worker": {"CPU": 2, "GPU": 0},
                "heartbeat_timeout": 30
            }
        """
        self.config = config
        self.logger = logging.getLogger("ray_head")
        self.workers = {}
        self.tasks = defaultdict(dict)
        self.health_monitor_running = False
        
        # Configuración de logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def start_cluster(self):
        """Inicia el cluster Ray como nodo head"""
        try:
            ray.init(
                address=self.config.get("head_host", "auto"),
                _redis_password=self.config.get("redis_password", "password"),
                dashboard_host='0.0.0.0',
                dashboard_port=self.config["dashboard_port"],
                include_dashboard=True,
                ignore_reinit_error=True
            )
            self.logger.info(f"Cluster Ray iniciado. Dashboard en http://localhost:{self.config['dashboard_port']}")
            
            # Iniciar monitoreo de salud
            self.health_monitor_running = True
            monitor_thread = Thread(target=self.monitor_cluster_health)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Error iniciando cluster: {str(e)}")
            raise
    
    def register_worker(self, worker_id: str, resources: Dict[str, Any]):
        """
        Registra un nuevo worker en el cluster
        
        Args:
            worker_id: Identificador único del worker
            resources: Recursos que reporta el worker
        """
        self.workers[worker_id] = {
            "status": "active",
            "last_heartbeat": time.time(),
            "resources": resources,
            "tasks_assigned": 0
        }
        self.logger.info(f"Worker registrado: {worker_id} con recursos: {resources}")
    
    def distribute_training_task(self, task_config: Dict) -> str:
        """
        Distribuye tareas de entrenamiento entre workers
        
        Args:
            task_config: {
                "task_id": str,
                "model_config": Dict,
                "data_partition": Any,
                "priority": int
            }
        
        Returns:
            str: ID de la tarea asignada
        """
        task_id = task_config.get("task_id", f"task_{len(self.tasks)+1}")
        
        # Encontrar worker disponible
        available_workers = [
            wid for wid, w in self.workers.items()
            if w["status"] == "active" 
            and w["tasks_assigned"] < self.config["max_tasks_per_worker"]
        ]
        
        if not available_workers:
            raise RuntimeError("No hay workers disponibles")
            
        worker_id = available_workers[0]
        
        try:
            # Usar el remote function directamente del worker
            worker = ray.get_actor(worker_id)
            future = worker.train_model.remote(
                task_config["model_config"],
                task_config["data_partition"]
            )
            
            self.tasks[task_id] = {
                "worker_id": worker_id,
                "future": future,
                "status": "pending",
                "start_time": time.time()
            }
            
            self.workers[worker_id]["tasks_assigned"] += 1
            self.logger.info(f"Tarea {task_id} asignada a worker {worker_id}")
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error asignando tarea {task_id}: {str(e)}")
            raise
    
    def monitor_cluster_health(self):
        """Monitorea la salud del cluster continuamente"""
        while self.health_monitor_running:
            try:
                current_time = time.time()
                dead_workers = []
                
                for worker_id, worker in self.workers.items():
                    # Verificar heartbeat
                    if (current_time - worker["last_heartbeat"]) > self.config["heartbeat_timeout"]:
                        self.logger.warning(f"Worker {worker_id} no responde")
                        worker["status"] = "unresponsive"
                        dead_workers.append(worker_id)
                    
                    # Rebalancear tareas si es necesario
                    self._rebalance_tasks()
                
                # Manejar workers muertos
                for worker_id in dead_workers:
                    self.handle_worker_failure(worker_id)
                
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error en monitoreo: {str(e)}")
                time.sleep(10)
    
    def _rebalance_tasks(self):
        """Reasigna tareas para balancear carga"""
        avg_tasks = sum(w["tasks_assigned"] for w in self.workers.values()) / max(1, len(self.workers))
        
        for worker_id, worker in self.workers.items():
            if worker["tasks_assigned"] > avg_tasks + 1:
                # Implementar lógica de rebalanceo aquí
                pass
    
    def handle_worker_failure(self, worker_id: str):
        """Maneja la falla de un worker redistribuyendo tareas"""
        if worker_id not in self.workers:
            return
            
        self.logger.warning(f"Manejando falla del worker {worker_id}")
        
        # Reasignar tareas pendientes
        tasks_to_redistribute = [
            task_id for task_id, task in self.tasks.items()
            if task["worker_id"] == worker_id and task["status"] == "pending"
        ]
        
        for task_id in tasks_to_redistribute:
            task_config = {
                "task_id": task_id,
                "model_config": self.tasks[task_id]["model_config"],
                "data_partition": self.tasks[task_id]["data_partition"]
            }
            self.distribute_training_task(task_config)
        
        # Eliminar worker
        del self.workers[worker_id]
        self.logger.info(f"Worker {worker_id} removido del cluster")
    
    def get_cluster_stats(self) -> Dict:
        """Obtiene estadísticas del cluster"""
        return {
            "total_workers": len(self.workers),
            "active_workers": sum(1 for w in self.workers.values() if w["status"] == "active"),
            "total_tasks": len(self.tasks),
            "pending_tasks": sum(1 for t in self.tasks.values() if t["status"] == "pending"),
            "completed_tasks": sum(1 for t in self.tasks.values() if t["status"] == "completed"),
            "resources": ray.cluster_resources(),
            "available_resources": ray.available_resources()
        }
    
    def shutdown_cluster(self):
        """Apaga el cluster de forma limpia"""
        self.health_monitor_running = False
        try:
            ray.shutdown()
            self.logger.info("Cluster Ray apagado correctamente")
        except Exception as e:
            self.logger.error(f"Error apagando cluster: {str(e)}")

def main():
    """Función principal que inicia el nodo head"""
    config = {
        "head_host": "0.0.0.0",
        "redis_port": 6379,
        "dashboard_port": 8265,
        "min_workers": 2,
        "max_workers": 10,
        "max_tasks_per_worker": 3,
        "heartbeat_timeout": 30,
        "resources_per_worker": {"CPU": 2}
    }
    
    head_node = RayHeadNode(config)
    head_node.start_cluster()
    
    try:
        # Mantener el nodo activo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        head_node.shutdown_cluster()

if __name__ == "__main__":
    main()
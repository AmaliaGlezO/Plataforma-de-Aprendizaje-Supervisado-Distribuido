"""
NODOS TRABAJADORES RAY
Este archivo maneja los workers que ejecutan las tareas de entrenamiento.
Funciones: conectar al head, ejecutar entrenamientos, reportar estado.
"""

import ray
import time
import uuid
import logging
from typing import Dict, Any
from threading import Thread

class RayWorkerNode:
    """Gestiona un nodo worker del cluster Ray"""
    
    def __init__(self, head_host: str, worker_id: str = None):
        """
        Inicializa worker y se conecta al nodo head
        
        Args:
            head_host: Dirección del nodo head
            worker_id: ID único para el worker (opcional)
        """
        self.head_host = head_host
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"ray_worker_{self.worker_id}")
        self.heartbeat_active = False
        self.current_tasks = 0
        
        # Configuración de logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def connect_to_head(self):
        """Se conecta al nodo head del cluster"""
        try:
            ray.init(
                address=self.head_host,
                runtime_env={"working_dir": "."},
                ignore_reinit_error=True
            )
            
            # Registrarse como actor
            ray.remote(RayWorkerNode).options(name=self.worker_id, get_if_exists=True).remote(
                self.head_host, self.worker_id
            )
            
            self.logger.info(f"Worker {self.worker_id} conectado a head {self.head_host}")
            return True
        except Exception as e:
            self.logger.error(f"Error conectando a head: {str(e)}")
            raise
    
    def register_with_cluster(self):
        """Se registra en el cluster para recibir tareas"""
        try:
            # Obtener referencia al head
            head = ray.get_actor("ray_head")
            
            # Reportar recursos
            resources = {
                "CPU": ray.state.cluster_resources().get("CPU", 0),
                "GPU": ray.state.cluster_resources().get("GPU", 0)
            }
            
            ray.get(head.register_worker.remote(self.worker_id, resources))
            
            # Iniciar heartbeat
            self.heartbeat_active = True
            heartbeat_thread = Thread(target=self._send_heartbeat)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            
            self.logger.info(f"Worker {self.worker_id} registrado en cluster")
            return True
        except Exception as e:
            self.logger.error(f"Error registrando worker: {str(e)}")
            raise
    
    def _send_heartbeat(self):
        """Envía señales de vida periódicas al head"""
        head = ray.get_actor("ray_head")
        while self.heartbeat_active:
            try:
                ray.get(head.report_heartbeat.remote(self.worker_id))
                time.sleep(5)
            except Exception as e:
                self.logger.warning(f"Error enviando heartbeat: {str(e)}")
                time.sleep(10)
    
    @ray.remote
    def train_model(self, model_config: Dict, data_partition: Any) -> Dict:
        """
        Entrena un modelo con los datos asignados
        
        Args:
            model_config: Configuración del modelo
            data_partition: Partición de datos asignada
            
        Returns:
            Dict: Resultados del entrenamiento
        """
        from sklearn.metrics import accuracy_score
        from ..ml_engine.model_factory import ModelFactory
        
        self.current_tasks += 1
        task_id = f"{self.worker_id}_task_{self.current_tasks}"
        self.logger.info(f"Iniciando tarea {task_id}")
        
        try:
            # Crear y entrenar modelo
            model = ModelFactory().create_model(
                model_config["model_type"],
                **model_config.get("params", {})
            )
            
            X, y = data_partition
            model.fit(X, y)
            
            # Calcular métricas simples
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            return {
                "status": "completed",
                "task_id": task_id,
                "worker_id": self.worker_id,
                "accuracy": accuracy,
                "model_type": model_config["model_type"]
            }
            
        except Exception as e:
            self.logger.error(f"Error en tarea {task_id}: {str(e)}")
            return {
                "status": "failed",
                "task_id": task_id,
                "error": str(e)
            }
        finally:
            self.current_tasks -= 1
    
    def cleanup_resources(self):
        """Libera recursos después de completar tareas"""
        # Implementar según necesidades específicas
        pass
    
    def shutdown(self):
        """Apaga el worker de forma limpia"""
        self.heartbeat_active = False
        try:
            ray.shutdown()
            self.logger.info(f"Worker {self.worker_id} apagado correctamente")
        except Exception as e:
            self.logger.error(f"Error apagando worker: {str(e)}")

def main():
    """Función principal que inicia el worker"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-host", required=True, help="Dirección del nodo head")
    parser.add_argument("--worker-id", help="ID único para el worker")
    args = parser.parse_args()
    
    worker = RayWorkerNode(args.head_host, args.worker_id)
    worker.connect_to_head()
    worker.register_with_cluster()
    
    try:
        # Mantener el worker activo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.shutdown()

if __name__ == "__main__":
    main()
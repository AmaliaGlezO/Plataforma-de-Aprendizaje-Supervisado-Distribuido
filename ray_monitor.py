import requests
import pandas as pd
import json
import time
from typing import Dict, Any, Optional
import logging
import ray

class RayMonitor:
    def __init__(self, ray_head_url="http://172.25.0.2:8265", timeout=10):
        self.base_url = ray_head_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def test_connection(self) -> bool:
        """Prueba la conexión básica al dashboard de Ray"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return ray.is_initialized()
        except Exception as e:
            self.logger.error(f"Error de conexión: {e}")
            return False
    
    def obtener_estado_cluster() -> Dict:
        """Obtiene el estado actual del cluster Ray con caché"""
        if not ray.is_initialized():
            return {
                "status": "Ray no inicializado",
                "alive_nodes": 0,
                "total_nodes": 0,
                "resources": {},
                "node_details": []
            }
        
        try:
            # Obtener información del cluster
            resources = ray.cluster_resources()
            nodes = ray.nodes()
            
            alive_nodes = [node for node in nodes if node.get('Alive', False)]
            dead_nodes = [node for node in nodes if not node.get('Alive', False)]
            
            # Procesar información de nodos
            node_details = []
            for node in alive_nodes:
                node_resources = node.get('Resources', {})
                node_details.append({
                    "node_id": node.get('NodeID', 'Desconocido'),
                    "node_ip": node.get('NodeManagerAddress', 'Desconocido'),
                    "cpu_available": int(node_resources.get('CPU', 0)),
                    "cpu_used": int(node_resources.get('CPU', 0)) - int(ray.available_resources().get('CPU', 0)),
                    "memory_available_gb": round(node_resources.get('memory', 0) / (1024**3), 2),
                    "memory_used_gb": round((node_resources.get('memory', 0) - ray.available_resources().get('memory', 0)) / (1024**3), 2),
                    "gpu_available": int(node_resources.get('GPU', 0)),
                    "object_store_memory_gb": round(node_resources.get('object_store_memory', 0) / (1024**3), 2),
                    "is_head": node.get('is_head_node', False)
                })
            
            return {
                "status": "Activo",
                "alive_nodes": len(alive_nodes),
                "total_nodes": len(nodes),
                "dead_nodes": len(dead_nodes),
                "total_cpu": int(resources.get('CPU', 0)),
                "total_memory_gb": round(resources.get('memory', 0) / (1024**3), 2),
                "available_cpu": int(ray.available_resources().get('CPU', 0)),
                "available_memory_gb": round(ray.available_resources().get('memory', 0) / (1024**3), 2),
                "resources": resources,
                "node_details": node_details,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"Error obteniendo estado del cluster: {str(e)}")
            return {
                "status": f"Error: {str(e)}",
                "alive_nodes": 0,
                "total_nodes": 0,
                "resources": {},
                "node_details": []
            }

        
    def get_nodes_data(self) -> pd.DataFrame:
        """Obtiene datos de los nodos en formato DataFrame"""
        status = self.get_cluster_status()
        
        if not status:
            return pd.DataFrame()
        
        # Diferentes formatos de respuesta de Ray
        nodes_data = []
        
        # Formato 1: nodos directos
        if "nodes" in status:
            for node_id, info in status["nodes"].items():
                nodes_data.append(self._parse_node_info(node_id, info))
        
        # Formato 2: estructura anidada
        elif "cluster" in status and "nodes" in status["cluster"]:
            for node_info in status["cluster"]["nodes"]:
                node_id = node_info.get("NodeID", "unknown")
                nodes_data.append(self._parse_node_info(node_id, node_info))
        
        # Formato 3: lista de nodos
        elif isinstance(status, list):
            for i, node_info in enumerate(status):
                node_id = node_info.get("NodeID", f"node_{i}")
                nodes_data.append(self._parse_node_info(node_id, node_info))
        
        return pd.DataFrame(nodes_data) if nodes_data else pd.DataFrame()
    
    def _parse_node_info(self, node_id: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Parsea información de un nodo individual"""
        return {
            "Node ID": node_id[:12] + "..." if len(node_id) > 15 else node_id,
            "State": info.get("state", info.get("State", "unknown")),
            "CPU Cores": info.get("Resources", {}).get("CPU", info.get("cpu", 0)),
            "Memory (GB)": round(info.get("Resources", {}).get("memory", info.get("memory", 0)) / (1024**3), 2),
            "GPU": info.get("Resources", {}).get("GPU", info.get("gpu", 0)),
            "IP": info.get("NodeManagerAddress", info.get("ip", "unknown")),
            "Uptime": self._format_uptime(info.get("uptime", 0))
        }
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Formatea el tiempo de actividad"""
        if not uptime_seconds:
            return "N/A"
        
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Obtiene resumen de uso de recursos"""
        status = self.get_cluster_status()
        nodes_df = self.get_nodes_data()
        
        if nodes_df.empty:
            return {}
        
        total_cpu = nodes_df["CPU Cores"].sum()
        total_memory = nodes_df["Memory (GB)"].sum()
        total_gpu = nodes_df["GPU"].sum()
        active_nodes = len(nodes_df[nodes_df["State"] == "ALIVE"])
        
        return {
            "total_nodes": len(nodes_df),
            "active_nodes": active_nodes,
            "total_cpu_cores": total_cpu,
            "total_memory_gb": round(total_memory, 2),
            "total_gpu": total_gpu,
            "cluster_utilization": f"{(active_nodes/len(nodes_df)*100):.1f}%" if len(nodes_df) > 0 else "0%"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica la salud del cluster"""
        health_status = {
            "connection": self.test_connection(),
            "api_accessible": bool(self.get_cluster_status()),
            "nodes_available": not self.get_nodes_data().empty,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        health_status["overall_healthy"] = all([
            health_status["connection"],
            health_status["api_accessible"],
            health_status["nodes_available"]
        ])
        
        return health_status

# Función auxiliar para diferentes URLs de Ray
def create_ray_monitor(custom_url: Optional[str] = None) -> RayMonitor:
    """Crea un monitor de Ray con diferentes URLs posibles"""
    possible_urls = [
        custom_url,
        "http://172.25.0.2:8265"
        "http://ray-head:8265",
        "http://localhost:8265",
        "http://127.0.0.1:8265",
        "http://ray-dashboard:8265"
    ]
    
    for url in possible_urls:
        if url is None:
            continue
            
        monitor = RayMonitor(url)
        if monitor.test_connection():
            print(f"✅ Conectado exitosamente a Ray en: {url}")
            return monitor
        else:
            print(f"❌ No se pudo conectar a: {url}")
    
    print("❌ No se pudo conectar a ninguna URL de Ray")
    return RayMonitor()  # Retorna monitor por defecto

def main():
    """Función principal para probar la conexión"""
    print("Iniciando monitor de Ray...")
    
    # Intentar conectar a diferentes URLs posibles
    possible_urls = [
        "http://ray-head:8265",  # Docker internal
        "http://localhost:8265",  # Localhost
        "http://127.0.0.1:8265"   # Alternative localhost
    ]
    
    connected = False
    for url in possible_urls:
        monitor = RayMonitor(url)
        if monitor.test_connection():
            print(f"\n✅ Conectado exitosamente a: {url}")
            monitor.obtener_estado_cluster()
            connected = True
            break
    
    if not connected:
        print("\n❌ No se pudo conectar a ninguna URL de Ray")
        print("Posibles soluciones:")
        print("1. Verifica que el servicio ray-head esté corriendo")
        print("2. Comprueba la red Docker: 'docker network inspect ray-network'")
        print("3. Prueba acceder manualmente al dashboard en tu navegador")

if __name__ == "__main__":
    main()
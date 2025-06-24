import streamlit as st
import ray
import os
import psutil
import pandas as pd
import plotly.graph_objects as go
import time
import subprocess
from typing import Dict, List, Optional

@st.cache_data(ttl=30)
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
        st.error(f"Error obteniendo estado del cluster: {str(e)}")
        return {
            "status": f"Error: {str(e)}",
            "alive_nodes": 0,
            "total_nodes": 0,
            "resources": {},
            "node_details": []
        }

@st.cache_data(ttl=10)
def obtener_metricas_sistema() -> Dict:
    """Obtiene métricas del sistema con caché"""
    try:
        # Obtener métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        # Obtener métricas de Ray si está disponible
        ray_metrics = {}
        if ray.is_initialized():
            try:
                ray_metrics = {
                    "tasks_running": len(ray.tasks()),
                    "objects_in_memory": len(ray.objects()),
                    "actors_running": len(ray.actors())
                }
            except Exception as e:
                ray_metrics = {"error": str(e)}
        
        # Obtener información de procesos Ray
        ray_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            if 'ray' in proc.info['name'].lower():
                ray_processes.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "cpu_percent": proc.info['cpu_percent'],
                    "memory_mb": round(proc.info['memory_info'].rss / (1024*1024), 2)
                })
        
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "per_cpu": psutil.cpu_percent(interval=1, percpu=True)
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent
            },
            "network": {
                "bytes_sent_mb": round(net_io.bytes_sent / (1024*1024), 2),
                "bytes_recv_mb": round(net_io.bytes_recv / (1024*1024), 2)
            },
            "ray": ray_metrics,
            "ray_processes": ray_processes
        }
    except Exception as e:
        st.error(f"Error obteniendo métricas del sistema: {str(e)}")
        return {
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

def graficar_metricas_cluster(estado_cluster: Dict):
    """Crea gráficos de métricas del cluster"""
    if not estado_cluster or "alive_nodes" not in estado_cluster:
        st.warning("No hay datos del cluster disponibles")
        return
    
    # Gráfico de uso de recursos
    fig_recursos = go.Figure()
    
    # CPU
    total_cpu = estado_cluster.get("total_cpu", 0)
    available_cpu = estado_cluster.get("available_cpu", 0)
    value = 100 * (total_cpu - available_cpu) / total_cpu if total_cpu > 0 else 0
    fig_recursos.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Uso de CPU"},
        domain={'x': [0, 0.3], 'y': [0.5, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))
    
    # Memoria
    total_memory_gb = estado_cluster.get("total_memory_gb", 0)
    available_memory_gb = estado_cluster.get("available_memory_gb", 0)
    value = 100 * (total_memory_gb - available_memory_gb) / total_memory_gb if total_memory_gb > 0 else 0
    fig_recursos.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Uso de Memoria"},
        domain={'x': [0.35, 0.65], 'y': [0.5, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ]
        }
    ))
    
    # Nodos
    fig_recursos.add_trace(go.Indicator(
        mode="number",
        value=estado_cluster["alive_nodes"],
        title={'text': "Nodos Activos"},
        domain={'x': [0.7, 1], 'y': [0.5, 1]},
        number={'suffix': f"/{estado_cluster['total_nodes']}", 'font': {'size': 40}}
    ))
    
    fig_recursos.update_layout(
        title="Resumen de Recursos del Cluster",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig_recursos, use_container_width=True)
    
    # Tabla de nodos detallada
    if estado_cluster.get("node_details"):
        st.subheader("Detalles de Nodos")
        df_nodes = pd.DataFrame(estado_cluster["node_details"])
        
        # Formatear columnas
        df_nodes["CPU"] = df_nodes.apply(
            lambda x: f"{x['cpu_used']}/{x['cpu_available']}", axis=1)
        df_nodes["Memoria (GB)"] = df_nodes.apply(
            lambda x: f"{x['memory_used_gb']:.1f}/{x['memory_available_gb']:.1f}", axis=1)
        df_nodes["Tipo"] = df_nodes["is_head"].apply(lambda x: "Head" if x else "Worker")
        
        # Mostrar tabla
        st.dataframe(
            df_nodes[["node_id", "node_ip", "Tipo", "CPU", "Memoria (GB)", "gpu_available"]],
            column_config={
                "node_id": "ID Nodo",
                "node_ip": "Dirección IP",
                "gpu_available": "GPUs Disponibles"
            },
            hide_index=True,
            use_container_width=True
        )

def renderizar_pestana_estado_cluster(estado_cluster: Dict, metricas_sistema: Dict):
    """Renderiza la pestaña de estado detallado del cluster"""
    st.header("Estado del Cluster Ray")
    
    if estado_cluster.get("status", "").startswith("Error"):
        st.error(f"Error obteniendo estado del cluster: {estado_cluster['status']}")
        return
    
    # Mostrar gráficos principales
    graficar_metricas_cluster(estado_cluster)
    
    # Mostrar métricas del sistema
    st.subheader("Métricas del Sistema")
    
    if metricas_sistema.get("error"):
        st.error(f"Error obteniendo métricas del sistema: {metricas_sistema['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Uso de CPU", f"{metricas_sistema['cpu']['percent']}%")
            st.metric("Núcleos CPU", metricas_sistema['cpu']['count'])
        
        with col2:
            st.metric("Uso de Memoria", f"{metricas_sistema['memory']['percent']}%")
            st.metric("Memoria Disponible", f"{metricas_sistema['memory']['available_gb']:.1f} GB")
        
        with col3:
            st.metric("Uso de Disco", f"{metricas_sistema['disk']['percent']}%")
            st.metric("Espacio Libre", f"{metricas_sistema['disk']['free_gb']:.1f} GB")
        
        # Gráfico de uso de CPU por núcleo
        if metricas_sistema['cpu'].get('per_cpu'):
            st.subheader("Uso de CPU por Núcleo")
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Bar(
                x=[f"Núcleo {i}" for i in range(len(metricas_sistema['cpu']['per_cpu']))],
                y=metricas_sistema['cpu']['per_cpu'],
                name="Uso de CPU"
            ))
            fig_cpu.update_layout(
                yaxis_title="Porcentaje de Uso",
                xaxis_title="Núcleos de CPU",
                height=300
            )
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        # Procesos Ray
        if metricas_sistema.get('ray_processes'):
            st.subheader("Procesos Ray en Ejecución")
            df_procesos = pd.DataFrame(metricas_sistema['ray_processes'])
            if not df_procesos.empty:
                st.dataframe(
                    df_procesos.sort_values('memory_mb', ascending=False),
                    column_config={
                        "pid": "PID",
                        "name": "Nombre",
                        "cpu_percent": "CPU %",
                        "memory_mb": "Memoria (MB)"
                    },
                    hide_index=True,
                    use_container_width=True
                )

def renderizar_pestana_metricas_sistema(metricas_sistema):
    # Verifica el contenido de metricas_sistema
    st.write(metricas_sistema)  # Para depurar y ver el contenido

    used_memory = metricas_sistema['memory'].get('used_gb', 0)  
    memory_gb = used_memory 

    # Resto de tu código para mostrar métricas

def añadir_worker_externo(nombre_worker: str, cpu_a_añadir: int) -> bool:
    """Añade un worker externo usando docker directamente"""
    try:
        if not nombre_worker or not cpu_a_añadir or cpu_a_añadir <= 0:
            st.error("Nombre de worker y CPUs deben ser valores válidos")
            return False
        
        # Obtener la dirección del head node del cluster Ray
        head_address = ray.get_runtime_context().gcs_address
        if not head_address:
            st.error("No se pudo obtener la dirección del head node")
            return False
        
        # Comando para iniciar un worker Ray en Docker
        comando = [
            "docker", "run", "-d",
            "--name", f"ray_worker_{nombre_worker}",
            "--cpus", str(cpu_a_añadir),
            "-e", f"RAY_HEAD_SERVICE_HOST={head_address.split(':')[0]}",
            "-e", f"RAY_HEAD_SERVICE_PORT={head_address.split(':')[1] if ':' in head_address else '6379'}",
            "rayproject/ray"
        ]
        
        # Ejecutar el comando
        result = subprocess.run(comando, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success(f"Worker {nombre_worker} añadido exitosamente con {cpu_a_añadir} CPUs")
            return True
        else:
            st.error(f"Error añadiendo worker: {result.stderr}")
            return False
            
    except Exception as e:
        st.error(f"Error al añadir worker: {str(e)}")
        return False

def eliminar_nodo_ray(nombre_nodo: str) -> bool:
    """Elimina un nodo Ray usando su nombre"""
    try:
        if not nombre_nodo:
            st.error("Nombre de nodo no válido")
            return False
        
        # Obtener todos los nodos
        nodes = ray.nodes()
        node_to_remove = None
        
        # Buscar el nodo por dirección IP o ID
        for node in nodes:
            if (nombre_nodo == node.get('NodeID') or 
                nombre_nodo == node.get('NodeManagerAddress')):
                node_to_remove = node
                break
        
        if not node_to_remove:
            st.error(f"No se encontró el nodo {nombre_nodo}")
            return False
        
        # No permitir eliminar el head node
        if node_to_remove.get('is_head_node', False):
            st.error("No se puede eliminar el head node del cluster")
            return False
        
        # Si es un worker en Docker
        if nombre_nodo.startswith("ray_worker_"):
            comando = ["docker", "rm", "-f", nombre_nodo]
            result = subprocess.run(comando, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success(f"Nodo {nombre_nodo} eliminado exitosamente")
                return True
            else:
                st.error(f"Error eliminando nodo: {result.stderr}")
                return False
        else:
            # Para otros tipos de nodos (no implementado completamente)
            st.warning("Eliminación de nodos no Docker no está completamente implementada")
            return False
            
    except Exception as e:
        st.error(f"Error eliminando nodo: {str(e)}")
        return False

def obtener_todos_los_nodos_ray() -> List[Dict]:
    """Obtiene la lista de todos los nodos Ray ejecutándose actualmente"""
    try:
        if not ray.is_initialized():
            return []
        
        nodes = ray.nodes()
        node_list = []
        
        for node in nodes:
            resources = node.get('Resources', {})
            node_list.append({
                "node_id": node.get('NodeID', 'Desconocido'),
                "node_ip": node.get('NodeManagerAddress', 'Desconocido'),
                "is_head": node.get('is_head_node', False),
                "is_alive": node.get('Alive', False),
                "cpu_available": int(resources.get('CPU', 0)),
                "memory_available_gb": round(resources.get('memory', 0) / (1024**3), 2),
                "gpu_available": int(resources.get('GPU', 0)),
                "object_store_memory_gb": round(resources.get('object_store_memory', 0) / (1024**3), 2)
            })
        
        return node_list
        
    except Exception as e:
        st.error(f"Error obteniendo nodos: {str(e)}")
        return []

if __name__ == '__main__':
    estado_cluster = obtener_estado_cluster()
    metricas_sistema = obtener_metricas_sistema()
    renderizar_pestana_estado_cluster(estado_cluster, metricas_sistema)
    st.write("---")
    st.write("Estado del clúster y métricas del sistema mostrados.")
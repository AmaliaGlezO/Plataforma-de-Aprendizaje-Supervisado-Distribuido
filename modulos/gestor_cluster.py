import streamlit as st
import ray
import os
import psutil
import pandas as pd
import plotly.graph_objects as go
import time
import subprocess

@st.cache_data(ttl=30)
def obtener_estado_cluster():
    """Obtiene el estado actual del cluster Ray con caché"""
    pass

@st.cache_data(ttl=10)
def obtener_metricas_sistema():
    """Obtiene métricas del sistema con caché"""
    pass

def graficar_metricas_cluster(estado_cluster):
    """Crea gráficos de métricas del cluster"""
    pass

def renderizar_pestana_estado_cluster(estado_cluster, metricas_sistema):
    """Renderiza la pestaña de estado detallado del cluster"""
    pass

def añadir_worker_externo(nombre_worker, cpu_a_añadir):
    """Añade un worker externo usando docker directamente"""
    pass

def eliminar_nodo_ray(nombre_nodo):
    """Elimina un nodo Ray usando su nombre"""
    pass

def obtener_todos_los_nodos_ray():
    """Obtiene la lista de todos los nodos Ray ejecutándose actualmente"""
    pass
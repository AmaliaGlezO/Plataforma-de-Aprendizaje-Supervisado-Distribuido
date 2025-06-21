"""
Ray ML Cluster Dashboard
"""

import streamlit as st
import time
from datetime import datetime

from modulos.utiles import initialize_session_state, load_custom_styles, get_unique_key
from modulos.gestor_cluster import obtener_estado_cluster, obtener_metricas_sistema, renderizar_pestana_estado_cluster
from modulos.interfaz_web import (renderizar_pestana_entrenamiento, renderizar_pestana_metricas_sistema)
from modulos.servidor_api import renderizar_pestana_api

# Configuración de la página debe ser la primera llamada de Streamlit
st.set_page_config(
    page_title="Ray ML Cluster Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.ray.io/',
        'About': '### Ray ML Cluster Dashboard\nSistema para entrenamiento distribuido de modelos ML'
    }
)

load_custom_styles()
initialize_session_state()

st.markdown("""
<div class="dashboard-header">
    <h1>🚀 Ray ML Cluster Dashboard</h1>
    <h3>Sistema de Entrenamiento ML Distribuido</h3>
</div>
""", unsafe_allow_html=True)


tab_titles = [
    "🔍 Estado del Cluster",
    "🧠 Entrenamiento ML",
    "🌐 API de Modelos",
    "💻 Métricas del Sistema",
]

tabs = st.tabs(tab_titles)

cluster_status = obtener_estado_cluster()
system_metrics = obtener_metricas_sistema()


with st.sidebar:
    st.title("⚙️ Configuración")
    
    auto_refresh = st.toggle(
        "Auto-Refresh (10s)",
        value=st.session_state.auto_refresh,
        key="auto_refresh_toggle",
    )
    
    if auto_refresh:
        st.session_state.auto_refresh = True
        time.sleep(0.1)  
        st.rerun()
    else:
        st.session_state.auto_refresh = False
    
    if st.button("🔄 Actualizar Ahora", key=get_unique_key("refresh_button")):
        st.experimental_rerun()
    
    

with tabs[0]:
    renderizar_pestana_estado_cluster(cluster_status,system_metrics)

with tabs[1]:
    renderizar_pestana_entrenamiento(cluster_status)


with tabs[2]:
    renderizar_pestana_api()

with tabs[3]:
    renderizar_pestana_metricas_sistema(system_metrics)

if st.session_state.auto_refresh:
    st.rerun()
    time.sleep(10)
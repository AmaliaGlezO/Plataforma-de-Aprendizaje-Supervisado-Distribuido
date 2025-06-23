"""
Centro de Control ML Distribuido
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
    page_title="Centro de Control ML Distribuido",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.ray.io/',
        'About': '### Centro de Control ML Distribuido\nPlataforma avanzada para orquestación de Machine Learning'
    }
)

# Estilos personalizados con la nueva paleta de colores
def load_enhanced_styles():
    st.markdown("""
    <style>
    /* Paleta de colores personalizada */
    :root {
        --gris-oscuro: #403D39;
        --blanco-roto: #FFFCF2;
        --naranja-terracota: #EB5E28;
        --casi-negro: #252422;
        --verde-grisaceo: #7E8D85;
        --beige-claro: #D8D2C3;
        --beige-grisaceo: #B8B2A6;
        --beige-calido: #E0DAD1;
        --marron-claro: #A68A64;
        --verde-apagado: #6B705C;
        --naranja-suave: #D4A373;
    }

    /* Fondo principal */
    .main .block-container {
        background: linear-gradient(135deg, var(--blanco-roto) 0%, var(--beige-claro) 100%);
        padding: 2rem 1rem;
    }

    /* Header principal */
    .dashboard-header {
        background: linear-gradient(135deg, var(--casi-negro) 0%, var(--gris-oscuro) 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(37, 36, 34, 0.3);
        border: 2px solid var(--beige-grisaceo);
    }

    .dashboard-header h1 {
        color: var(--blanco-roto);
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    .dashboard-header h3 {
        color: var(--naranja-suave);
        font-size: 1.3rem;
        margin: 0;
        font-weight: 400;
    }

    /* Sidebar personalizada */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--gris-oscuro) 0%, var(--casi-negro) 100%);
    }

    .sidebar .sidebar-content {
        background: var(--gris-oscuro);
        color: var(--blanco-roto);
    }

    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--beige-calido);
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(64, 61, 57, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--gris-oscuro);
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: var(--beige-grisaceo);
        color: var(--casi-negro);
    }

    .stTabs [aria-selected="true"] {
        background: var(--naranja-terracota) !important;
        color: var(--blanco-roto) !important;
        box-shadow: 0 4px 12px rgba(235, 94, 40, 0.4);
    }

    /* Métricas y tarjetas */
    .metric-card {
        background: linear-gradient(135deg, var(--blanco-roto) 0%, var(--beige-claro) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid var(--beige-grisaceo);
        box-shadow: 0 6px 20px rgba(64, 61, 57, 0.15);
        margin-bottom: 1rem;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, var(--naranja-terracota) 0%, var(--naranja-suave) 100%);
        color: var(--blanco-roto);
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(235, 94, 40, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--naranja-suave) 0%, var(--marron-claro) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(235, 94, 40, 0.4);
    }

    /* Toggle switches */
    .stToggle > label {
        background: var(--verde-grisaceo);
        border-radius: 25px;
    }

    /* Alertas y notificaciones */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid var(--naranja-terracota);
        background: var(--beige-calido);
        color: var(--casi-negro);
    }

    /* Selectbox y inputs */
    .stSelectbox > div > div {
        background: var(--beige-claro);
        border: 2px solid var(--beige-grisaceo);
        border-radius: 10px;
        color: var(--casi-negro);
    }

    /* Texto general */
    .stMarkdown {
        color: var(--gris-oscuro);
    }

    /* Títulos de sección */
    .section-title {
        color: var(--casi-negro);
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--naranja-terracota);
    }

    /* Contenedores de contenido */
    .content-container {
        background: var(--blanco-roto);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid var(--beige-grisaceo);
        box-shadow: 0 4px 16px rgba(64, 61, 57, 0.1);
        margin-bottom: 1.5rem;
    }

    /* Status indicators */
    .status-active {
        color: var(--verde-apagado);
        font-weight: bold;
    }

    .status-warning {
        color: var(--naranja-terracota);
        font-weight: bold;
    }

    .status-error {
        color: var(--marron-claro);
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

load_enhanced_styles()
initialize_session_state()

st.markdown("""
<div class="dashboard-header">
    <h1>⚡ Centro de Control ML</h1>
    <h3>Plataforma de Orquestación Distribuida</h3>
</div>
""", unsafe_allow_html=True)

# Títulos renovados para las pestañas
tab_titles = [
    "🎯 Monitor del Cluster",
    "🔬 Laboratorio ML",
    "🛡️ Gateway de Modelos",
    "📊 Telemetría del Sistema",
]

tabs = st.tabs(tab_titles)

cluster_status = obtener_estado_cluster()
system_metrics = obtener_metricas_sistema()

# Sidebar con nuevo estilo
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #403D39 0%, #252422 100%); border-radius: 15px; margin-bottom: 1rem;">
        <h2 style="color: #FFFCF2; margin: 0;">⚙️ Centro de Control</h2>
    </div>
    """, unsafe_allow_html=True)
    
    auto_refresh = st.toggle(
        "🔄 Actualización Automática (10s)",
        value=st.session_state.auto_refresh,
        key="auto_refresh_toggle",
    )
    
    if auto_refresh:
        st.session_state.auto_refresh = True
        time.sleep(0.1)  
        st.rerun()
    else:
        st.session_state.auto_refresh = False
    
    if st.button("⚡ Sincronizar Ahora", key=get_unique_key("refresh_button")):
        st.experimental_rerun()
    
    # Información adicional en sidebar
    st.markdown("""
    <div class="content-container" style="margin-top: 2rem;">
        <h4 style="color: #252422;">📈 Estado General</h4>
        <p style="color: #403D39; margin: 0;">Sistema operativo y monitoreando recursos</p>
    </div>
    """, unsafe_allow_html=True)

# Contenido de las pestañas con nuevos títulos
with tabs[0]:
    st.markdown('<h2 class="section-title">🎯 Monitor del Cluster</h2>', unsafe_allow_html=True)
    renderizar_pestana_estado_cluster(cluster_status, system_metrics)

with tabs[1]:
    st.markdown('<h2 class="section-title">🔬 Laboratorio ML</h2>', unsafe_allow_html=True)
    renderizar_pestana_entrenamiento(cluster_status)

with tabs[2]:
    st.markdown('<h2 class="section-title">🛡️ Gateway de Modelos</h2>', unsafe_allow_html=True)
    renderizar_pestana_api()

with tabs[3]:
    st.markdown('<h2 class="section-title">📊 Telemetría del Sistema</h2>', unsafe_allow_html=True)
    renderizar_pestana_metricas_sistema(system_metrics)

# Auto-refresh logic
if st.session_state.auto_refresh:
    st.rerun()
    time.sleep(10)
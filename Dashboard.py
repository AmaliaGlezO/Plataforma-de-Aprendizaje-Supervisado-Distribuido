import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import time

# Importar los monitores mejorados
try:
    from ray_monitor import RayMonitor, create_ray_monitor
    from ray_diagnostics import diagnose_ray_connection
except ImportError:
    st.error("No se pudieron importar los módulos de Ray. Asegúrate de tener ray_monitor_improved.py y ray_diagnostics.py")
    st.stop()

# Configuración inicial
st.set_page_config(layout="wide", page_title="ML Dashboard", page_icon="🚀")

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.status-good { color: #28a745; font-weight: bold; }
.status-bad { color: #dc3545; font-weight: bold; }
.status-warning { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Estado de la sesión
if 'ray_monitor' not in st.session_state:
    st.session_state.ray_monitor = None
if 'ray_connected' not in st.session_state:
    st.session_state.ray_connected = False

# Funciones auxiliares
def load_training_results():
    try:
        with open('training_results/training_results.json') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("📁 No se encontraron resultados de entrenamiento")
        return {}

def render_metrics_comparison(results):
    if not results:
        st.info("📊 No hay resultados de entrenamiento para mostrar")
        return
        
    df = pd.DataFrame.from_dict(results, orient='index')
    st.header("📈 Comparación de Modelos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Métricas Clave")
        st.dataframe(df[['mse', 'mae', 'r2']].sort_values('mse'), use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Tiempo de Entrenamiento")
        fig = px.bar(df, x=df.index, y='training_time', 
                     labels={'index':'Modelo', 'training_time':'Tiempo (s)'},
                     color='training_time', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

def show_connection_diagnostics():
    """Muestra diagnósticos de conexión a Ray"""
    st.header("🔧 Diagnósticos de Conexión")
    
    with st.expander("🔍 Ejecutar Diagnóstico Completo", expanded=False):
        if st.button("Ejecutar Diagnóstico", type="primary"):
            with st.spinner("Ejecutando diagnósticos..."):
                # Capturar salida de diagnóstico
                import io
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()
                
                try:
                    diagnose_ray_connection()
                    output = captured_output.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                st.text(output)

def show_ray_dashboard():
    st.header("🖥️ Monitoreo del Cluster Ray")
    
    # Configuración de conexión
    with st.sidebar:
        st.subheader("⚙️ Configuración Ray")
        ray_url = st.text_input("URL de Ray Dashboard", value="http://ray-head:8265")
        
        if st.button("🔄 Reconectar"):
            st.session_state.ray_monitor = RayMonitor(ray_url)
            st.session_state.ray_connected = False
    
    # Inicializar monitor si no existe
    if st.session_state.ray_monitor is None:
        st.session_state.ray_monitor = create_ray_monitor(ray_url)
    
    monitor = st.session_state.ray_monitor
    
    # Verificar salud del cluster
    health = monitor.health_check()
    
    # Mostrar estado de conexión
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_class = "status-good" if health["connection"] else "status-bad"
        st.markdown(f'<p class="{status_class}">🔗 Conexión: {"✅" if health["connection"] else "❌"}</p>', 
                   unsafe_allow_html=True)
    
    with col2:
        status_class = "status-good" if health["api_accessible"] else "status-bad"
        st.markdown(f'<p class="{status_class}">🔌 API: {"✅" if health["api_accessible"] else "❌"}</p>', 
                   unsafe_allow_html=True)
    
    with col3:
        status_class = "status-good" if health["nodes_available"] else "status-bad"
        st.markdown(f'<p class="{status_class}">🖥️ Nodos: {"✅" if health["nodes_available"] else "❌"}</p>', 
                   unsafe_allow_html=True)
    
    with col4:
        status_class = "status-good" if health["overall_healthy"] else "status-bad"
        st.markdown(f'<p class="{status_class}">🏥 Estado: {"Sano" if health["overall_healthy"] else "Problema"}</p>', 
                   unsafe_allow_html=True)
    
    # Si no hay conexión, mostrar diagnósticos
    if not health["overall_healthy"]:
        st.error("❌ No se pudo conectar al cluster de Ray")
        show_connection_diagnostics()
        return
    
    # Dashboard principal
    st.session_state.ray_connected = True
    
    # Resumen de recursos
    resource_usage = monitor.get_resource_usage()
    if resource_usage:
        st.subheader("📊 Resumen de Recursos")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🖥️ Nodos Totales", resource_usage.get("total_nodes", 0))
        with col2:
            st.metric("✅ Nodos Activos", resource_usage.get("active_nodes", 0))
        with col3:
            st.metric("🔧 CPU Cores", resource_usage.get("total_cpu_cores", 0))
        with col4:
            st.metric("💾 Memoria (GB)", resource_usage.get("total_memory_gb", 0))
    
    # Tabla de nodos
    st.subheader("🖥️ Nodos del Cluster")
    nodes_df = monitor.get_nodes_data()
    
    if not nodes_df.empty:
        st.dataframe(nodes_df, use_container_width=True, hide_index=True)
        
        # Gráficos de recursos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Distribución de Memoria")
            fig = px.pie(nodes_df, values='Memory (GB)', names='Node ID', 
                        title="Memoria por Nodo")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔧 Distribución de CPU")
            fig = px.bar(nodes_df, x='Node ID', y='CPU Cores', 
                        color='State', title="CPU Cores por Nodo")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No se pudieron obtener datos de los nodos")

def show_predictions_interface():
    """Interfaz para hacer predicciones"""
    st.header("🔮 Interfaz de Predicciones")
    
    # Verificar si Ray está conectado
    if not st.session_state.ray_connected:
        st.warning("⚠️ Necesitas conectar a Ray primero para hacer predicciones")
        return
    
    st.info("🚧 Implementar interfaz de predicciones aquí")
    
    # Ejemplo de interfaz
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Entrada de Datos")
        uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Vista previa de datos:")
            st.dataframe(df.head())
    
    with col2:
        st.subheader("⚙️ Configuración del Modelo")
        model_type = st.selectbox("Tipo de Modelo", 
                                 ["Linear Regression", "Random Forest", "XGBoost"])
        batch_size = st.slider("Tamaño de lote", 1, 1000, 100)
        
        if st.button("🚀 Ejecutar Predicción"):
            st.success("✅ Predicción ejecutada (simulada)")

def main():
    st.title("🚀 Panel de Control de ML Distribuido")
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Modelos", "🖥️ Monitoreo Ray", "🔮 Predicciones", "🔧 Diagnósticos"])
    
    with tab1:
        results = load_training_results()
        render_metrics_comparison(results)
    
    with tab2:
        show_ray_dashboard()
    
    with tab3:
        show_predictions_interface()
    
    with tab4:
        show_connection_diagnostics()
    
    # Footer
    st.markdown("---")
    st.markdown("🕐 Última actualización: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
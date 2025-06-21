import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from entrenador import EntrenamientoDistribuido
import time
import pickle
import ray
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import logging

# Configuración de la página
st.set_page_config(
    page_title="Entrenador ML Distribuido",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cargar_resultados_entrenamiento():
    """Carga los resultados de entrenamiento guardados"""
    try:
        if os.path.exists("training_results.json"):
            with open("training_results.json", 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error cargando resultados: {e}")
        return {}

def graficar_comparacion_modelos(datos_resultados, prefijo_grafico="default"):
    """Crea gráfico de comparación de modelos"""
    if not datos_resultados:
        st.warning("No hay datos de resultados disponibles")
        return
    
    # Crear DataFrame con los resultados
    df_results = []
    for modelo, datos in datos_resultados.items():
        if datos.get('status') == 'success':
            df_results.append({
                'Modelo': modelo,
                'MSE': datos.get('mse', 0),
                'MAE': datos.get('mae', 0),
                'R²': datos.get('r2', 0),
                'Tiempo (s)': datos.get('training_time', 0),
                'CV MSE': datos.get('cv_mean', 0)
            })
    
    if not df_results:
        st.warning("No hay modelos exitosos para graficar")
        return
    
    df = pd.DataFrame(df_results)
    
    # Gráfico de barras para MSE
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Comparación MSE")
        fig_mse = px.bar(
            df.sort_values('MSE'), 
            x='Modelo', 
            y='MSE',
            title=f"Mean Squared Error por Modelo ({prefijo_grafico})",
            color='MSE',
            color_continuous_scale='Viridis_r'
        )
        fig_mse.update_xaxis(tickangle=45)
        st.plotly_chart(fig_mse, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Comparación R²")
        fig_r2 = px.bar(
            df.sort_values('R²', ascending=False), 
            x='Modelo', 
            y='R²',
            title=f"R² Score por Modelo ({prefijo_grafico})",
            color='R²',
            color_continuous_scale='Viridis'
        )
        fig_r2.update_xaxis(tickangle=45)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Gráfico de dispersión tiempo vs performance
    st.subheader("⚡ Tiempo vs Performance")
    fig_scatter = px.scatter(
        df, 
        x='Tiempo (s)', 
        y='R²',
        size='MSE',
        hover_data=['MAE', 'CV MSE'],
        text='Modelo',
        title=f"Tiempo de Entrenamiento vs R² ({prefijo_grafico})"
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter, use_container_width=True)

def ejecutar_entrenamiento_distribuido(nombre_dataset, modelos_seleccionados, habilitar_tolerancia_fallos=True):
    """Ejecuta entrenamiento distribuido con tolerancia a fallos"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🚀 Inicializando entrenador distribuido...")
        progress_bar.progress(10)
        
        # Inicializar entrenador
        trainer = EntrenamientoDistribuido(enable_fault_tolerance=habilitar_tolerancia_fallos)
        
        status_text.text("📊 Obteniendo información del cluster...")
        progress_bar.progress(20)
        
        # Mostrar información del cluster
        cluster_info = trainer.get_cluster_info()
        st.info(f"Recursos del cluster: {cluster_info}")
        
        status_text.text("🤖 Iniciando entrenamiento de modelos...")
        progress_bar.progress(30)
        
        # Ejecutar entrenamiento
        start_time = time.time()
        resultados = trainer.train_models_distributed(selected_models=modelos_seleccionados)
        end_time = time.time()
        
        progress_bar.progress(80)
        status_text.text("💾 Guardando resultados...")
        
        # Guardar resultados
        trainer.save_results(f"training_results_{nombre_dataset}.json")
        trainer.save_models(f"models_{nombre_dataset}")
        
        progress_bar.progress(100)
        status_text.text("✅ Entrenamiento completado!")
        
        # Mostrar estadísticas
        fault_stats = trainer.get_fault_tolerance_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Tiempo Total", f"{end_time - start_time:.2f}s")
        with col2:
            st.metric("🎯 Modelos Exitosos", len([r for r in resultados.values() if r.get('status') == 'success']))
        with col3:
            st.metric("❌ Modelos Fallidos", fault_stats['failed_tasks'])
        with col4:
            st.metric("🖥️ Nodos Vivos", fault_stats['alive_nodes'])
        
        return resultados, fault_stats
        
    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")
        return {}, {}

def ejecutar_entrenamiento_secuencial(lista_datasets, modelos_seleccionados):
    """Ejecuta entrenamiento secuencial de múltiples datasets"""
    todos_resultados = {}
    resumen_ejecucion = {
        'datasets_procesados': [],
        'timestamp': datetime.now().isoformat(),
        'modelos_utilizados': modelos_seleccionados,
        'estadisticas_globales': {}
    }
    
    progress_container = st.container()
    
    for i, dataset in enumerate(lista_datasets):
        with progress_container:
            st.subheader(f"📊 Procesando Dataset: {dataset}")
            
            # Ejecutar entrenamiento para este dataset
            resultados, fault_stats = ejecutar_entrenamiento_distribuido(
                dataset, modelos_seleccionados
            )
            
            if resultados:
                todos_resultados[dataset] = resultados
                resumen_ejecucion['datasets_procesados'].append({
                    'nombre': dataset,
                    'modelos_exitosos': len([r for r in resultados.values() if r.get('status') == 'success']),
                    'modelos_fallidos': fault_stats.get('failed_tasks', 0),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Mostrar resultados del dataset actual
                st.success(f"✅ Completado: {dataset}")
                graficar_comparacion_modelos(resultados, f"Dataset: {dataset}")
            else:
                st.error(f"❌ Error procesando: {dataset}")
    
    # Guardar resumen de ejecución
    with open("resumen_ejecucion_secuencial.json", 'w') as f:
        json.dump(resumen_ejecucion, f, indent=2)
    
    return todos_resultados, resumen_ejecucion

def cargar_resumen_ejecucion():
    """Carga el resumen de ejecución secuencial"""
    try:
        if os.path.exists("resumen_ejecucion_secuencial.json"):
            with open("resumen_ejecucion_secuencial.json", 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error cargando resumen: {e}")
        return {}

def graficar_comparacion_cruzada(todos_resultados):
    """Crea gráfico de comparación entre datasets"""
    if not todos_resultados:
        st.warning("No hay datos para comparar")
        return
    
    # Preparar datos para comparación cruzada
    comparison_data = []
    
    for dataset, resultados in todos_resultados.items():
        for modelo, datos in resultados.items():
            if datos.get('status') == 'success':
                comparison_data.append({
                    'Dataset': dataset,
                    'Modelo': modelo,
                    'MSE': datos.get('mse', 0),
                    'MAE': datos.get('mae', 0),
                    'R²': datos.get('r2', 0),
                    'Tiempo': datos.get('training_time', 0)
                })
    
    if not comparison_data:
        st.warning("No hay datos exitosos para comparar")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Heatmap de MSE por Dataset y Modelo
    st.subheader("🔥 Heatmap de MSE: Datasets vs Modelos")
    pivot_mse = df_comparison.pivot(index='Dataset', columns='Modelo', values='MSE')
    
    fig_heatmap = px.imshow(
        pivot_mse,
        title="MSE por Dataset y Modelo",
        color_continuous_scale='Viridis_r',
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Gráfico de líneas para comparar performance
    st.subheader("📈 Comparación de R² entre Datasets")
    fig_lines = px.line(
        df_comparison,
        x='Modelo',
        y='R²',
        color='Dataset',
        title="R² Score por Modelo y Dataset",
        markers=True
    )
    fig_lines.update_xaxis(tickangle=45)
    st.plotly_chart(fig_lines, use_container_width=True)

def obtener_estadisticas_tolerancia_fallos():
    """Obtiene estadísticas de tolerancia a fallos del entrenador"""
    try:
        # Inicializar entrenador para obtener estadísticas
        trainer = EntrenamientoDistribuido(enable_fault_tolerance=True)
        return trainer.get_fault_tolerance_stats()
    except Exception as e:
        st.error(f"Error obteniendo estadísticas: {e}")
        return {}

def graficar_metricas_entrenamiento(historial_entrenamiento, prefijo_grafico=""):
    """Visualiza métricas de rendimiento de los modelos"""
    if not historial_entrenamiento:
        st.warning("No hay historial de entrenamiento disponible")
        return
    
    # Crear gráfico de métricas en el tiempo
    timestamps = [entry['timestamp'] for entry in historial_entrenamiento]
    modelos = list(set([entry['modelo'] for entry in historial_entrenamiento]))
    
    fig = go.Figure()
    
    for modelo in modelos:
        datos_modelo = [entry for entry in historial_entrenamiento if entry['modelo'] == modelo]
        mse_values = [entry['mse'] for entry in datos_modelo]
        timestamps_modelo = [entry['timestamp'] for entry in datos_modelo]
        
        fig.add_trace(go.Scatter(
            x=timestamps_modelo,
            y=mse_values,
            mode='lines+markers',
            name=f'{modelo} MSE',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f"Evolución de MSE en el Tiempo {prefijo_grafico}",
        xaxis_title="Timestamp",
        yaxis_title="MSE",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def ejecutar_entrenamiento_distribuido_avanzado(nombre_dataset, modelos_seleccionados, hiperparametros=None, habilitar_tolerancia_fallos=True, callback_progreso=None):
    """Ejecuta entrenamiento distribuido avanzado con monitoreo en tiempo real"""
    # Contenedor para métricas en tiempo real
    metrics_container = st.container()
    progress_container = st.container()
    
    # Placeholder para logs en tiempo real
    log_placeholder = st.empty()
    
    try:
        with progress_container:
            st.info("🚀 Iniciando entrenamiento avanzado...")
            
        # Inicializar entrenador
        trainer = EntrenamientoDistribuido(enable_fault_tolerance=habilitar_tolerancia_fallos)
        
        # Configurar hiperparámetros si se proporcionan
        if hiperparametros:
            st.info(f"🔧 Aplicando hiperparámetros: {hiperparametros}")
        
        # Ejecutar entrenamiento con monitoreo
        start_time = time.time()
        
        # Simular progreso en tiempo real
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(len(modelos_seleccionados)):
            progress = (i + 1) / len(modelos_seleccionados)
            progress_bar.progress(progress)
            status_text.text(f"Entrenando modelo {i+1}/{len(modelos_seleccionados)}")
            
            if callback_progreso:
                callback_progreso(progress, f"Modelo {i+1}")
        
        # Ejecutar entrenamiento real
        resultados = trainer.train_models_distributed(selected_models=modelos_seleccionados)
        
        end_time = time.time()
        
        # Mostrar métricas finales
        with metrics_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Tiempo Total", f"{end_time - start_time:.2f}s")
            with col2:
                exitosos = len([r for r in resultados.values() if r.get('status') == 'success'])
                st.metric("✅ Exitosos", exitosos)
            with col3:
                fallidos = len([r for r in resultados.values() if r.get('status') == 'failed'])
                st.metric("❌ Fallidos", fallidos)
        
        return resultados
        
    except Exception as e:
        st.error(f"Error en entrenamiento avanzado: {e}")
        return {}

def cargar_historial_entrenamiento(nombre_dataset):
    """Carga el historial de entrenamiento guardado"""
    archivo_historial = f"historial_{nombre_dataset}.json"
    try:
        if os.path.exists(archivo_historial):
            with open(archivo_historial, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error cargando historial: {e}")
        return []

def cargar_modelo_entrenado(nombre_dataset, nombre_modelo):
    """Carga un modelo entrenado desde el sistema de archivos"""
    archivo_modelo = f"models_{nombre_dataset}/{nombre_modelo}.pkl"
    try:
        if os.path.exists(archivo_modelo):
            with open(archivo_modelo, 'rb') as f:
                return pickle.load(f)
        else:
            st.error(f"Modelo no encontrado: {archivo_modelo}")
            return None
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None

def obtener_lista_modelos_entrenados(nombre_dataset):
    """Obtiene la lista de modelos entrenados disponibles para un dataset"""
    directorio_modelos = f"models_{nombre_dataset}"
    modelos_disponibles = []
    
    try:
        if os.path.exists(directorio_modelos):
            archivos = os.listdir(directorio_modelos)
            modelos_disponibles = [
                archivo.replace('.pkl', '') 
                for archivo in archivos 
                if archivo.endswith('.pkl')
            ]
        return modelos_disponibles
    except Exception as e:
        st.error(f"Error listando modelos: {e}")
        return []

def graficar_metricas_inferencia(datos_inferencia, prefijo_grafico=""):
    """Visualiza métricas de inferencia de los modelos entrenados"""
    if not datos_inferencia:
        st.warning("No hay datos de inferencia disponibles")
        return
    
    # Crear gráficos de predicciones vs valores reales
    fig_pred = go.Figure()
    
    for modelo, datos in datos_inferencia.items():
        if 'predictions' in datos and 'y_true' in datos:
            fig_pred.add_trace(go.Scatter(
                x=datos['y_true'],
                y=datos['predictions'],
                mode='markers',
                name=modelo,
                opacity=0.6
            ))
    
    # Línea de predicción perfecta
    if datos_inferencia:
        sample_data = list(datos_inferencia.values())[0]
        if 'y_true' in sample_data:
            min_val = min(sample_data['y_true'])
            max_val = max(sample_data['y_true'])
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predicción Perfecta',
                line=dict(dash='dash', color='red')
            ))
    
    fig_pred.update_layout(
        title=f"Predicciones vs Valores Reales {prefijo_grafico}",
        xaxis_title="Valores Reales",
        yaxis_title="Predicciones"
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)

# INTERFAZ PRINCIPAL DE STREAMLIT
def main():
    st.title("🤖 Entrenador ML Distribuido")
    st.markdown("### Sistema de entrenamiento distribuido con tolerancia a fallos")
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Selección de modo de operación
    modo = st.sidebar.selectbox(
        "Modo de Operación",
        ["Entrenamiento Individual", "Entrenamiento Secuencial", "Análisis de Resultados", "Inferencia"]
    )
    
    # Lista de modelos disponibles
    modelos_disponibles = [
        'RandomForest', 'GradientBoosting', 'LinearRegression', 'SVR', 'KNN',
        'XGBoost', 'Ridge', 'Lasso', 'AdaBoost', 'ExtraTrees', 'DecisionTree',
        'SGD', 'PassiveAggressive', 'LinearSVR', 'MLP', 'Bagging', 'Voting'
    ]
    
    if modo == "Entrenamiento Individual":
        st.header("🎯 Entrenamiento Individual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input("Nombre del Dataset", "energia_dataset")
            tolerancia_fallos = st.checkbox("Habilitar Tolerancia a Fallos", True)
        
        with col2:
            modelos_seleccionados = st.multiselect(
                "Seleccionar Modelos",
                modelos_disponibles,
                default=['RandomForest', 'GradientBoosting', 'LinearRegression', 'SVR']
            )
        
        if st.button("🚀 Iniciar Entrenamiento", type="primary"):
            if modelos_seleccionados:
                with st.spinner("Entrenando modelos..."):
                    resultados, fault_stats = ejecutar_entrenamiento_distribuido(
                        dataset_name, modelos_seleccionados, tolerancia_fallos
                    )
                
                if resultados:
                    st.success("✅ Entrenamiento completado!")
                    
                    # Mostrar resultados
                    st.subheader("📊 Resultados del Entrenamiento")
                    graficar_comparacion_modelos(resultados, dataset_name)
                    
                    # Mostrar tabla de resultados
                    st.subheader("📋 Tabla de Resultados")
                    df_results = []
                    for modelo, datos in resultados.items():
                        if datos.get('status') == 'success':
                            df_results.append({
                                'Modelo': modelo,
                                'MSE': f"{datos.get('mse', 0):.4f}",
                                'MAE': f"{datos.get('mae', 0):.4f}",
                                'R²': f"{datos.get('r2', 0):.4f}",
                                'CV MSE': f"{datos.get('cv_mean', 0):.4f} ± {datos.get('cv_std', 0):.4f}",
                                'Tiempo (s)': f"{datos.get('training_time', 0):.2f}"
                            })
                    
                    if df_results:
                        st.dataframe(pd.DataFrame(df_results), use_container_width=True)
                else:
                    st.error("❌ Error durante el entrenamiento")
            else:
                st.warning("⚠️ Selecciona al menos un modelo")
    
    elif modo == "Entrenamiento Secuencial":
        st.header("🔄 Entrenamiento Secuencial")
        
        datasets_input = st.text_area(
            "Datasets (uno por línea)",
            "energia_dataset\nventas_dataset\nproduccion_dataset"
        )
        
        datasets_list = [d.strip() for d in datasets_input.split('\n') if d.strip()]
        
        modelos_seleccionados = st.multiselect(
            "Seleccionar Modelos",
            modelos_disponibles,
            default=['RandomForest', 'GradientBoosting', 'LinearRegression']
        )
        
        if st.button("🔄 Iniciar Entrenamiento Secuencial", type="primary"):
            if datasets_list and modelos_seleccionados:
                with st.spinner("Ejecutando entrenamiento secuencial..."):
                    todos_resultados, resumen = ejecutar_entrenamiento_secuencial(
                        datasets_list, modelos_seleccionados
                    )
                
                if todos_resultados:
                    st.success("✅ Entrenamiento secuencial completado!")
                    
                    # Mostrar comparación cruzada
                    st.subheader("🔀 Comparación Cruzada")
                    graficar_comparacion_cruzada(todos_resultados)
                    
                    # Mostrar resumen
                    st.subheader("📊 Resumen de Ejecución")
                    st.json(resumen)
            else:
                st.warning("⚠️ Especifica datasets y modelos")
    
    elif modo == "Análisis de Resultados":
        st.header("📈 Análisis de Resultados")
        
        # Cargar y mostrar resultados existentes
        resultados = cargar_resultados_entrenamiento()
        
        if resultados:
            st.subheader("📊 Resultados Cargados")
            graficar_comparacion_modelos(resultados, "Análisis")
            
            # Estadísticas de tolerancia a fallos
            st.subheader("🛡️ Estadísticas de Tolerancia a Fallos")
            fault_stats = obtener_estadisticas_tolerancia_fallos()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🖥️ Nodos en Cluster", fault_stats.get('cluster_nodes', 0))
            with col2:
                st.metric("✅ Nodos Vivos", fault_stats.get('alive_nodes', 0))
            with col3:
                st.metric("❌ Tareas Fallidas", fault_stats.get('failed_tasks', 0))
            
            # Mostrar detalles de fallos si existen
            if fault_stats.get('failed_task_details'):
                st.subheader("🔍 Detalles de Tareas Fallidas")
                for failed_task in fault_stats['failed_task_details']:
                    st.error(f"**{failed_task.get('model_name', 'Desconocido')}**: {failed_task.get('error', 'Error desconocido')}")
        else:
            st.info("No hay resultados de entrenamiento disponibles. Ejecuta un entrenamiento primero.")
    
    elif modo == "Inferencia":
        st.header("🔮 Inferencia con Modelos Entrenados")
        
        dataset_name = st.selectbox(
            "Seleccionar Dataset",
            ["energia_dataset", "ventas_dataset", "produccion_dataset"]
        )
        
        modelos_disponibles_dataset = obtener_lista_modelos_entrenados(dataset_name)
        
        if modelos_disponibles_dataset:
            modelo_seleccionado = st.selectbox(
                "Seleccionar Modelo",
                modelos_disponibles_dataset
            )
            
            if st.button("🔮 Cargar Modelo", type="primary"):
                modelo = cargar_modelo_entrenado(dataset_name, modelo_seleccionado)
                
                if modelo:
                    st.success(f"✅ Modelo {modelo_seleccionado} cargado correctamente!")
                    st.info("Modelo listo para inferencia. Aquí puedes agregar la lógica para hacer predicciones.")
                    
                    # Aquí podrías agregar campos de entrada para hacer predicciones
                    st.subheader("📝 Hacer Predicción")
                    st.info("Implementa aquí los campos de entrada según tu dataset")
        else:
            st.warning(f"No hay modelos entrenados disponibles para {dataset_name}")
    
    # Footer con información del sistema
    st.markdown("---")
    st.markdown("**🤖 Entrenador ML Distribuido** - Sistema con tolerancia a fallos usando Ray")
    
    # Mostrar estado de Ray si está inicializado
    if ray.is_initialized():
        st.success("✅ Ray inicializado y conectado")
        cluster_resources = ray.cluster_resources()
        st.text(f"Recursos disponibles: {cluster_resources}")
    else:
        st.info("ℹ️ Ray no inicializado")

if __name__ == "__main__":
    main()
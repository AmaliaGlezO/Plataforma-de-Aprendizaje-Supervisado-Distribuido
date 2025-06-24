import streamlit as st
import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
import plotly.express as px

def obtener_url_base_api():
    if os.environ.get('API_BASE_URL'):
        return os.environ.get('API_BASE_URL')
    elif os.path.exists('/.dockerenv') or os.environ.get('CONTAINER_NAME'):
        return "http://ml-api:8000"
    else:
        return "http://localhost:8000"

URL_BASE_API = obtener_url_base_api()

# Helper functions
def make_get_request(endpoint):
    try:
        response = requests.get(f"{URL_BASE_API}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making request to {endpoint}: {str(e)}")
        return None

def make_post_request(endpoint, data=None, files=None):
    try:
        if files:
            response = requests.post(f"{URL_BASE_API}{endpoint}", files=files)
        else:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{URL_BASE_API}{endpoint}", 
                json=data,
                headers=headers
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making request to {endpoint}: {str(e)}")
        return None

def display_model_info(model_info):
    st.subheader("Model Information")
    st.json(model_info)

def display_cluster_status(status):
    st.subheader("Cluster Status")
    
    cols = st.columns(4)
    cols[0].metric("Total Nodes", status['total_nodes'])
    cols[1].metric("Alive Nodes", status['alive_nodes'])
    cols[2].metric("Total CPUs", status['total_cpus'])
    cols[3].metric("Total Memory (GB)", status['total_memory_gb'])
    
    st.write("### Node Details")
    st.dataframe(pd.DataFrame(status['node_details']))

def display_inference_stats(stats):
    st.subheader("Inference Statistics")
    
    cols = st.columns(4)
    cols[0].metric("Total Predictions", stats['total_predictions'])
    cols[1].metric("Avg Prediction Time (s)", f"{stats['avg_prediction_time']:.4f}")
    cols[2].metric("Errors", stats['error_count'])
    cols[3].metric("Uptime (hours)", f"{stats['uptime_hours']:.2f}")
    
    if stats['model_usage']:
        st.write("### Model Usage")
        usage_df = pd.DataFrame.from_dict(stats['model_usage'], orient='index', columns=['Count'])
        st.dataframe(usage_df.sort_values('Count', ascending=False))
        
        fig = px.pie(usage_df, values='Count', names=usage_df.index, title='Model Usage Distribution')
        st.plotly_chart(fig)

def display_training_results(results):
    st.subheader("Training Results")
    
    if isinstance(results, dict) and 'results' in results:
        summary = results.copy()
        results_data = summary.pop('results')
        
        st.write("### Summary")
        cols = st.columns(4)
        cols[0].metric("Total Models", summary['total_models'])
        cols[1].metric("Successful Models", summary['successful_models'])
        cols[2].metric("Regression Models", summary['regression_models'])
        cols[3].metric("Classification Models", summary['classification_models'])
        
        st.write("### Detailed Results")
        results_df = pd.DataFrame.from_dict(results_data, orient='index')
        st.dataframe(results_df)
    else:
        st.write(results)

def display_prediction_results(result, is_regression):
    st.success("Prediction successful!")
    st.write("### Prediction Result")
    
    # Mostrar resultados en formato adecuado
    if is_regression:
        cols = st.columns(3)
        cols[0].metric("Predicted Value", f"{result['predictions'][0]:.2f}")
        cols[1].metric("Prediction Time", f"{result['prediction_time']:.4f}s")
        cols[2].metric("Features Used", result['feature_count'])
    else:
        cols = st.columns(4)
        cols[0].metric("Predicted Class", result['predictions'][0])
        cols[1].metric("Confidence", 
                      f"{max(result['probabilities'][0])*100:.1f}%" if result.get('probabilities') else "N/A")
        cols[2].metric("Prediction Time", f"{result['prediction_time']:.4f}s")
        cols[3].metric("Is High", "Yes" if result['predictions'][0] == 1 else "No")
    
    # Mostrar probabilidades si es clasificación
    if result.get('probabilities'):
        st.write("### Class Probabilities")
        prob_df = pd.DataFrame(
            result['probabilities'],
            columns=[f"Class {i}" for i in range(len(result['probabilities'][0]))]
        )
        st.dataframe(prob_df.style.format("{:.2%}"))

# Main app
def main():
    st.set_page_config(page_title="Distributed ML API Dashboard", layout="wide")
    st.title("Distributed ML API Dashboard")
    
    # Health check
    with st.expander("System Health"):
        health = make_get_request("/health")
        if health:
            cols = st.columns(4)
            cols[0].metric("Status", health['status'])
            cols[1].metric("Models Loaded", health['models_loaded'])
            cols[2].metric("Cluster Nodes", health['cluster_nodes'])
            cols[3].metric("Uptime (hours)", f"{health['uptime_hours']:.2f}")
            
            st.write(f"Last check: {health['timestamp']}")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Models", "Training", "Prediction", 
        "Cluster Info", "Advanced"
    ])
    
    # Models tab
    with tab1:
        st.header("Model Management")
        
        # List models
        if st.button("Refresh Models"):
            st.session_state.models = make_get_request("/models")
        
        if 'models' not in st.session_state:
            st.session_state.models = make_get_request("/models")
        
        if st.session_state.models:
            st.write(f"Found {len(st.session_state.models)} models")
            
            # Model selection
            model_names = [m['name'] for m in st.session_state.models]
            selected_model = st.selectbox("Select a model", model_names)
            
            # Model details
            if selected_model:
                model_details = make_get_request(f"/models/{selected_model}")
                if model_details:
                    display_model_info(model_details)
                    
                    # Delete button
                    if st.button("Delete Model"):
                        result = requests.delete(f"{URL_BASE_API}/models/{selected_model}")
                        if result.status_code == 200:
                            st.success(f"Model {selected_model} deleted successfully")
                            st.session_state.models = make_get_request("/models")
                        else:
                            st.error(f"Error deleting model: {result.text}")
        
        # Model search
        st.subheader("Search Models")
        search_query = st.text_input("Enter model name or algorithm")
        if search_query:
            search_results = make_get_request(f"/models/search/{search_query}")
            if search_results:
                st.dataframe(pd.DataFrame(search_results))
    
    # Training tab
    with tab2:
        st.header("Model Training")
        
        with st.form("training_form"):
            st.write("Configure training parameters")
            
            task_type = st.selectbox(
                "Task Type",
                ["regression", "classification", "both"],
                index=2
            )
            
            test_size = st.slider(
                "Test Size Ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05
            )
            
            # Get available algorithms
            algorithms = make_get_request("/algorithms")
            if algorithms:
                all_algorithms = []
                if algorithms['algorithms']['regression']:
                    all_algorithms.extend([
                        f"Regression: {name}" 
                        for name in algorithms['algorithms']['regression']
                    ])
                if algorithms['algorithms']['classification']:
                    all_algorithms.extend([
                        f"Classification: {name}" 
                        for name in algorithms['algorithms']['classification']
                    ])
                
                selected_algorithms = st.multiselect(
                    "Select algorithms (leave empty for all)",
                    all_algorithms
                )
                
                # Extract just the algorithm names
                selected_models = [
                    alg.split(": ")[1] 
                    for alg in selected_algorithms
                ] if selected_algorithms else None
            
            submit_train = st.form_submit_button("Start Training")
            
            if submit_train:
                with st.spinner("Starting training job..."):
                    training_request = {
                        "task_type": task_type,
                        "test_size": test_size,
                        "selected_models": selected_models
                    }
                    
                    response = make_post_request("/train", training_request)
                    if response:
                        st.success("Training started in background!")
                        st.json(response)
        
        # Training results
        if st.button("View Training Results"):
            results = make_get_request("/training/results")
            if results:
                display_training_results(results)
    
    # Prediction tab
    with tab3:
        st.header("Model Prediction")
        
        pred_tab1, pred_tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
       # En la sección de Prediction tab (dentro de pred_tab1), reemplazar el código actual con:

    with pred_tab1:
        st.subheader("Single Prediction")
        
        if 'models' in st.session_state and st.session_state.models:
            model_names = [m['name'] for m in st.session_state.models]
            selected_model = st.selectbox(
                "Select model for prediction",
                model_names,
                key="pred_model_select"
            )
            
            # Feature input
            st.write("Enter feature values:")
            
            if selected_model:
                model_details = make_get_request(f"/models/{selected_model}")
                if model_details:
                    is_regression = "_REG" in selected_model
                    is_classification = "_CLF" in selected_model
                    
                    # Definición completa de 18 características
                    feature_config = [
                        {"name": "year", "desc": "Año (2020-2025)", "default": 2023, "type": int},
                        {"name": "month_num", "desc": "Mes (1-12)", "default": 6, "type": int},
                        {"name": "day", "desc": "Día del mes (1-31)", "default": 15, "type": int},
                        {"name": "hour", "desc": "Hora (0-23)", "default": 14, "type": int},
                        {"name": "day_of_week", "desc": "Día semana (0=lunes)", "default": 2, "type": int},
                        {"name": "is_weekend", "desc": "Fin de semana (0=no, 1=sí)", "default": 0, "type": int},
                        {"name": "disponibilidad", "desc": "Disponibilidad energética (MW)", "default": 5000.0, "type": float},
                        {"name": "demanda_maxima", "desc": "Demanda máxima (MW)", "default": 5500.0, "type": float},
                        {"name": "deficit", "desc": "Déficit actual (MW)", "default": 0.0, "type": float},
                        {"name": "capacidad_utilizada", "desc": "Capacidad utilizada (ratio)", "default": 0.909, "type": float},
                        {"name": "deficit_ratio", "desc": "Ratio de déficit", "default": 0.0, "type": float},
                        {"name": "respaldo_ratio", "desc": "Ratio de respaldo", "default": 0.2, "type": float},
                        {"name": "temp_ambiente", "desc": "Temperatura ambiente (°C)", "default": 25.5, "type": float},
                        {"name": "humedad", "desc": "Humedad relativa (%)", "default": 60.0, "type": float},
                        {"name": "precio_energia", "desc": "Precio energía (€/MWh)", "default": 45.30, "type": float},
                        {"name": "indice_consumo", "desc": "Índice de consumo", "default": 1.2, "type": float},
                        {"name": "categoria_horaria", "desc": "Categoría horaria (1-3)", "default": 2, "type": int},
                        {"name": "estacion", "desc": "Estación del año (1-4)", "default": 3, "type": int}
                    ]
                    
                    # Mostrar campos de entrada organizados
                    features = []
                    cols = st.columns(3)
                    for i, feat in enumerate(feature_config):
                        with cols[i % 3]:
                            feature = st.number_input(
                                feat["desc"],
                                value=feat["default"],
                                key=f"feature_{i}",
                                step=1.0 if feat["type"] == float else 1
                            )
                            features.append(feature)
                    
                    # Configuración de probabilidades
                    return_proba = st.checkbox(
                        "Return probabilities (classification only)",
                        value=False,
                        disabled=is_regression
                    )
                    
                    # Botón de predicción
                    if st.button("Predict"):
                        prediction_request = {
                            "model_name": selected_model,
                            "features": [features],
                            "return_probabilities": return_proba
                        }
                        
                        with st.spinner("Making prediction..."):
                            result = make_post_request("/predict", prediction_request)
                            if result:
                                display_prediction_results(result, is_regression)
            
        with pred_tab2:
            st.subheader("Batch Prediction")
            
            if 'models' in st.session_state and st.session_state.models:
                model_names = [m['name'] for m in st.session_state.models]
                selected_model = st.selectbox(
                    "Select model for batch prediction",
                    model_names,
                    key="batch_model_select"
                )
                
                return_proba = st.checkbox(
                    "Return probabilities (classification only)",
                    value=False,
                    key="batch_proba"
                )
                
                uploaded_file = st.file_uploader(
                    "Upload CSV file with features",
                    type=["csv"]
                )
                
                if uploaded_file is not None and selected_model:
                    if st.button("Predict Batch"):
                        with st.spinner("Processing batch prediction..."):
                            # Create proper multipart form data
                            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                            data = {
                                'model_name': selected_model,
                                'return_probabilities': str(return_proba).lower()
                            }
                            
                            response = requests.post(
                                f"{URL_BASE_API}/predict/batch",
                                files=files,
                                data=data
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("Batch prediction completed!")
                                
                                # Display results
                                st.write("### Predictions")
                                predictions_df = pd.DataFrame({
                                    "Predictions": result['predictions']
                                })
                                st.dataframe(predictions_df)
                                
                                if 'probabilities' in result and result['probabilities']:
                                    st.write("### Probabilities")
                                    prob_df = pd.DataFrame(result['probabilities'])
                                    st.dataframe(prob_df)
                                
                                st.write("### Metrics")
                                cols = st.columns(2)
                                cols[0].metric("Batch Size", result['batch_size'])
                                cols[1].metric("Prediction Time (s)", 
                                             f"{result['batch_prediction_time']:.4f}")
                            else:
                                st.error(f"Error in batch prediction: {response.text}")
    
    # Cluster Info tab
    with tab4:
        st.header("Cluster Information")
        
        if st.button("Refresh Cluster Status"):
            status = make_get_request("/cluster/status")
            if status:
                display_cluster_status(status)
        
        if 'cluster_status' not in st.session_state:
            st.session_state.cluster_status = make_get_request("/cluster/status")
        
        if st.session_state.cluster_status:
            display_cluster_status(st.session_state.cluster_status)
        
        st.subheader("Inference Statistics")
        stats = make_get_request("/inference-stats")
        if stats:
            display_inference_stats(stats)
    
    # Advanced tab
    with tab5:
        st.header("Advanced Operations")
        
        if st.button("View Available Algorithms"):
            algorithms = make_get_request("/algorithms")
            if algorithms:
                st.write(f"Total algorithms: {algorithms['total_algorithms']}")
                
                st.write("### Regression Algorithms")
                reg_df = pd.DataFrame.from_dict(
                    algorithms['algorithms']['regression'], 
                    orient='index'
                )
                st.dataframe(reg_df)
                
                st.write("### Classification Algorithms")
                clf_df = pd.DataFrame.from_dict(
                    algorithms['algorithms']['classification'], 
                    orient='index'
                )
                st.dataframe(clf_df)
        
        if st.button("View Raw Health Check"):
            health = make_get_request("/health")
            if health:
                st.json(health)

if __name__ == "__main__":
    main()
import ray
import logging
import time
import uuid
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models import create_model_instance, evaluate_model, serialize_model_for_ray, deserialize_model_from_ray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_ray_cluster(address='auto'):
    """Inicializa el cluster Ray con manejo de errores"""
    try:
        if not ray.is_initialized():
            ray.init(address=address)
            logger.info("✅ Ray cluster initialized successfully")
            logger.info(f"Cluster resources: {ray.cluster_resources()}")
        else:
            logger.info("⚠️ Ray already initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        raise

def get_cluster_status():
    """Obtiene información detallada del cluster"""
    return {
        'nodes': len(ray.nodes()),
        'resources': ray.cluster_resources(),
        'available_resources': ray.available_resources(),
        'dashboard_url': ray.get_dashboard_url()
    }

def get_node_resources():
    """Obtiene información detallada por nodo"""
    return [{
        'node_id': node['NodeID'],
        'resources': node['Resources'],
        'alive': node['Alive']
    } for node in ray.nodes()]

@ray.remote
class ModelStore:
    """Actor para almacenar modelos y métricas de forma persistente"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.datasets = {}
        logger.info("📦 ModelStore initialized")
    
    def store_model(self, model_id, model, metrics, dataset_id=None):
        """Guarda modelo entrenado con sus métricas"""
        serialized_model = serialize_model_for_ray(model)
        self.models[model_id] = serialized_model
        self.metrics[model_id] = metrics
        if dataset_id:
            self.datasets[model_id] = dataset_id
        logger.info(f"💾 Model {model_id} stored")
        return True
    
    def get_model(self, model_id):
        """Recupera modelo por ID"""
        serialized = self.models.get(model_id)
        if serialized:
            return deserialize_model_from_ray(serialized)
        return None
    
    def get_all_models(self):
        """Obtiene lista de todos los modelos disponibles"""
        return list(self.models.keys())
    
    def get_metrics(self, model_id=None):
        """Obtiene métricas de un modelo específico o todos"""
        if model_id:
            return self.metrics.get(model_id)
        return self.metrics
    
    def delete_model(self, model_id):
        """Elimina modelo del almacén"""
        if model_id in self.models:
            del self.models[model_id]
            del self.metrics[model_id]
            if model_id in self.datasets:
                del self.datasets[model_id]
            logger.info(f"🗑️ Model {model_id} deleted")
            return True
        return False
    
    

@ray.remote
def train_single_model(data_ref, model_type, model_params=None, dataset_id=None):
    """
    Entrena un modelo individual de forma remota
    
    Args:
        data_ref: Referencia Ray a los datos (X, y)
        model_type: Tipo de modelo a entrenar
        model_params: Parámetros para el modelo
        dataset_id: ID del dataset para tracking
    
    Returns:
        dict: Información del modelo entrenado
    """
    try:
        start_time = time.time()
        X, y = ray.get(data_ref)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = create_model_instance(model_type, model_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics['training_time'] = time.time() - start_time
        
        # Generate unique ID
        model_id = f"{model_type}_{uuid.uuid4().hex[:8]}"
        
        return {
            'model_id': model_id,
            'model_type': model_type,
            'model': model,
            'metrics': metrics,
            'dataset_id': dataset_id,
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error training {model_type}: {str(e)}")
        return {
            'model_type': model_type,
            'status': 'failed',
            'error': str(e)
        }

# Versión modificada de parallel_train_models para múltiples datasets
@ray.remote
def parallel_train_multiple_datasets(datasets_config):
    """
    Entrena modelos en múltiples datasets simultáneamente
    
    Args:
        datasets_config: Lista de tuplas (data, model_configs, dataset_id)
    
    Returns:
        dict: Resultados por dataset
    """
    results = {}
    
    # Lanzar todos los entrenamientos
    futures = []
    for data, model_configs, dataset_id in datasets_config:
        data_ref = ray.put(data)
        for config in model_configs:
            futures.append(
                train_single_model.remote(
                    data_ref,
                    config['type'],
                    config.get('params'),
                    dataset_id
                )
            )
    
    # Esperar resultados
    all_results = ray.get(futures)
    
    # Organizar por dataset
    model_store = create_model_store()
    for result in all_results:
        if result['status'] == 'success':
            dataset_id = result['dataset_id']
            if dataset_id not in results:
                results[dataset_id] = []
            results[dataset_id].append(result)
            
            # Almacenar modelo
            ray.get(model_store.store_model.remote(
                result['model_id'],
                result['model'],
                result['metrics'],
                dataset_id
            ))
    
    return results

def create_model_store():
    """Crea o recupera el actor ModelStore"""
    try:
        return ray.get_actor("model_store")
    except ValueError:
        logger.info("Creating new ModelStore actor")
        return ModelStore.options(name="model_store", lifetime="detached").remote()

def parallel_train_models(data, model_configs, dataset_id=None):
    """
    Coordina el entrenamiento paralelo de múltiples modelos
    
    Args:
        data: Tupla de (X, y)
        model_configs: Lista de configuraciones de modelos
        dataset_id: ID del dataset para tracking
    
    Returns:
        Lista de resultados del entrenamiento
    """
    # Upload data to object store once
    data_ref = ray.put(data)
    
    # Launch training tasks
    futures = [
        train_single_model.remote(
            data_ref,
            config['type'],
            config.get('params'),
            dataset_id
        ) for config in model_configs
    ]
    
    # Wait for completion
    results = ray.get(futures)
    
    # Store successful models
    model_store = create_model_store()
    for result in results:
        if result['status'] == 'success':
            ray.get(model_store.store_model.remote(
                result['model_id'],
                result['model'],
                result['metrics'],
                result['dataset_id']
            ))
    
    return results
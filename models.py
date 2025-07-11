from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    'RandomForest': {
        'class': RandomForestClassifier,
        'default_params': {
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'LogisticRegression': {
        'class': LogisticRegression,
        'default_params': {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'SVC': {
        'class': SVC,
        'default_params': {
            'probability': True,
            'random_state': 42
        }
    },
    'KNeighbors': {
        'class': KNeighborsClassifier,
        'default_params': {
            'n_jobs': -1
        }
    },
    'DecisionTree': {
        'class': DecisionTreeClassifier,
        'default_params': {
            'random_state': 42
        }
    },
    'GradientBoosting': {
        'class': GradientBoostingClassifier,
        'default_params': {
            'random_state': 42
        }
    }
}

def get_model_configs():
    """Retorna configuraciones predefinidas de modelos"""
    return MODEL_CONFIGS

def create_model_instance(model_type, params=None):
    """
    Crea instancia de modelo según tipo y parámetros
    
    Args:
        model_type (str): Tipo de modelo (ej. 'RandomForest')
        params (dict): Parámetros para sobrescribir los defaults
    
    Returns:
        Modelo de scikit-learn configurado
    """
    config = MODEL_CONFIGS.get(model_type)
    if not config:
        raise ValueError(f"Model type {model_type} not supported")
    
    final_params = config['default_params'].copy()
    if params:
        final_params.update(params)
    
    logger.info(f"Creating {model_type} with params: {final_params}")
    return config['class'](**final_params)

def evaluate_model(model, X_test, y_test):
    """
    Evalúa modelo y retorna métricas
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
    
    Returns:
        dict: Métricas de evaluación
    """
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

def serialize_model_for_ray(model):
    """Serializa modelo para ser almacenado en Ray"""
    return pickle.dumps(model)

def deserialize_model_from_ray(model_data):
    """Deserializa modelo desde Ray object store"""
    return pickle.loads(model_data)


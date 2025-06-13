"""
FÁBRICA DE MODELOS
Este archivo crea y configura diferentes tipos de modelos ML.
Funciones: crear modelos, configurar hiperparámetros, validar configuraciones.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, Union, List
import inspect

class ModelFactory:
    """Fábrica para crear diferentes tipos de modelos ML"""
    
    def __init__(self):
        """Inicializa la fábrica con modelos disponibles"""
        self.available_models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC,
            'logistic_regression': LogisticRegression,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }
        
        # Parámetros recomendados para problemas de clasificación
        self.default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'logistic_regression': {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'objective': 'binary:logistic',
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': -1,
                'learning_rate': 0.05,
                'objective': 'binary',
                'random_state': 42
            }
        }
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        """
        Crea una instancia de modelo según el tipo especificado
        
        Args:
            model_type: Tipo de modelo a crear (ej: 'random_forest')
            **kwargs: Parámetros adicionales para el modelo
            
        Returns:
            Instancia del modelo configurado
            
        Raises:
            ValueError: Si el tipo de modelo no es soportado
        """
        model_class = self.available_models.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Modelo no soportado: {model_type}. Modelos disponibles: {list(self.available_models.keys())}")
        
        # Combinar parámetros por defecto con los proporcionados
        params = self.get_default_params(model_type)
        params.update(kwargs)
        
        return model_class(**params)
    
    def get_default_params(self, model_type: str) -> Dict:
        """
        Obtiene parámetros por defecto para un tipo de modelo
        
        Args:
            model_type: Tipo de modelo
            
        Returns:
            Diccionario con parámetros por defecto
            
        Raises:
            ValueError: Si el tipo de modelo no es soportado
        """
        if model_type.lower() not in self.available_models:
            raise ValueError(f"Modelo no soportado: {model_type}")
            
        return self.default_params.get(model_type.lower(), {}).copy()
    
    def validate_model_config(self, config: Dict) -> bool:
        """
        Valida que la configuración del modelo sea correcta
        
        Args:
            config: Diccionario con configuración del modelo
                   Debe contener 'model_type' y opcionalmente 'params'
                   
        Returns:
            bool: True si la configuración es válida
            
        Raises:
            ValueError: Con detalles de la validación fallida
        """
        if not isinstance(config, dict):
            raise ValueError("La configuración debe ser un diccionario")
            
        if 'model_type' not in config:
            raise ValueError("La configuración debe incluir 'model_type'")
            
        model_type = config['model_type']
        if model_type.lower() not in self.available_models:
            raise ValueError(f"Modelo no soportado: {model_type}")
            
        # Validar parámetros específicos
        if 'params' in config:
            model_class = self.available_models[model_type.lower()]
            valid_params = inspect.signature(model_class.__init__).parameters
            
            for param in config['params']:
                if param not in valid_params:
                    raise ValueError(f"Parámetro no válido '{param}' para modelo {model_type}")
                    
        return True
    
    def create_multiple_models(self, configs: List[Dict]) -> List[Any]:
        """
        Crea múltiples modelos con diferentes configuraciones
        
        Args:
            configs: Lista de diccionarios de configuración
                     Cada dict debe contener 'model_type' y opcionalmente 'params'
                     
        Returns:
            Lista de instancias de modelos
            
        Raises:
            ValueError: Si alguna configuración no es válida
        """
        models = []
        for config in configs:
            self.validate_model_config(config)
            model = self.create_model(
                config['model_type'],
                **(config.get('params', {}))
            )
            models.append(model)
            
        return models
    
    def get_model_info(self, model_type: str) -> Dict:
        """
        Obtiene información sobre un tipo de modelo
        
        Args:
            model_type: Tipo de modelo
            
        Returns:
            Diccionario con:
            - description: Descripción breve
            - params: Parámetros disponibles
            - default_params: Parámetros por defecto
            
        Raises:
            ValueError: Si el tipo de modelo no es soportado
        """
        if model_type.lower() not in self.available_models:
            raise ValueError(f"Modelo no soportado: {model_type}")
            
        model_class = self.available_models[model_type.lower()]
        params = inspect.signature(model_class.__init__).parameters
        
        return {
            'description': model_class.__doc__.split('\n')[0] if model_class.__doc__ else '',
            'params': list(params.keys()),
            'default_params': self.get_default_params(model_type)
        }
    
    def list_available_models(self) -> List[str]:
        """
        Lista todos los tipos de modelos disponibles
        
        Returns:
            Lista de nombres de modelos disponibles
        """
        return list(self.available_models.keys())
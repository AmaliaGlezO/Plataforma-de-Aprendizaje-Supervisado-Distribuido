import ray
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote
def train_model(data_path, model_type, hyperparams, model_name, output_dir='models/'):
    """
    Función remota de Ray para entrenar un modelo de ML en paralelo
    
    Args:
        data_path (str): Ruta al dataset CSV
        model_type (str): Tipo de modelo ('random_forest', 'svm', 'linear_regression')
        hyperparams (dict): Hiperparámetros específicos del modelo
        model_name (str): Nombre único para el modelo guardado
        output_dir (str): Directorio donde guardar el modelo
    
    Returns:
        dict: Resultados del entrenamiento (métricas y metadata)
    """
    
    try:
        logger.info(f"Iniciando entrenamiento de {model_name} con {model_type}")
        
        # Cargar datos
        df = pd.read_csv(data_path)
        
        # Preprocesamiento de datos energéticos
        df = preprocess_energy_data(df)
        
        # Definir features y target
        feature_columns = [
            'disponibilidad', 'demanda_maxima', 'afectacion', 'respaldo',
            'horario_pico', 'unidades_averia', 'unidades_mantenimiento',
            'limitacion_termica', 'motores_impacto', 'year', 'month'
        ]
        
        X = df[feature_columns]
        y = df['deficit']  # Variable objetivo: déficit de energía
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalado de features (importante para SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Selección y configuración del modelo
        if model_type == 'random_forest':
            model = RandomForestRegressor(**hyperparams, random_state=42)
            X_train_final = X_train
            X_test_final = X_test
        elif model_type == 'svm':
            model = SVR(**hyperparams)
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        elif model_type == 'linear_regression':
            model = LinearRegression(**hyperparams)
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info(f"Entrenando {model_name}...")
        
        model.fit(X_train_final, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Predicciones y métricas
        y_pred_train = model.predict(X_train_final)
        y_pred_test = model.predict(X_test_final)
        
        # Calcular métricas
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'training_time_seconds': training_time
        }
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar modelo y scaler
        model_path = os.path.join(output_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(output_dir, f"{model_name}_scaler.joblib")
        
        # Preparar objeto para guardar
        model_package = {
            'model': model,
            'scaler': scaler if model_type in ['svm', 'linear_regression'] else None,
            'feature_columns': feature_columns,
            'model_type': model_type,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"Modelo guardado en: {model_path}")
        
        # Preparar resultados
        results = {
            'model_name': model_name,
            'model_type': model_type,
            'model_path': model_path,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'training_time': training_time,
            'status': 'success'
        }
        
        logger.info(f"Entrenamiento completado: {model_name}")
        logger.info(f"Test R2: {metrics['test_r2']:.4f}, Test MAE: {metrics['test_mae']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error entrenando {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'model_type': model_type,
            'status': 'error',
            'error': str(e)
        }

def preprocess_energy_data(df):
    """
    Preprocesa los datos energéticos
    """
    # Convertir fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Mapear meses a números
    month_mapping = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    df['month'] = df['month'].str.lower().map(month_mapping).fillna(df['month'])
    
    # Limpiar datos faltantes
    df = df.dropna()
    
    # Asegurar tipos de datos correctos
    numeric_columns = [
        'disponibilidad', 'demanda_maxima', 'afectacion', 'deficit',
        'respaldo', 'horario_pico', 'unidades_averia', 'unidades_mantenimiento',
        'limitacion_termica', 'motores_impacto', 'year', 'month'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    return df

def get_model_configurations():
    """
    Define las configuraciones de modelos e hiperparámetros para entrenar
    """
    configurations = [
        {
            'model_type': 'random_forest',
            'hyperparams': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            'model_name': 'rf_baseline'
        },
        {
            'model_type': 'random_forest',
            'hyperparams': {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2},
            'model_name': 'rf_deep'
        },
        {
            'model_type': 'random_forest',
            'hyperparams': {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10},
            'model_name': 'rf_shallow'
        },
        {
            'model_type': 'svm',
            'hyperparams': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
            'model_name': 'svm_rbf'
        },
        {
            'model_type': 'svm',
            'hyperparams': {'kernel': 'linear', 'C': 0.5, 'epsilon': 0.01},
            'model_name': 'svm_linear'
        },
        {
            'model_type': 'linear_regression',
            'hyperparams': {},
            'model_name': 'linear_baseline'
        }
    ]
    
    return configurations
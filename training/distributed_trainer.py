import ray
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import pandas as pd

# Configuración de modelos e hiperparámetros
MODELS_CONFIG = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 5}
    },
    "svm": {
        "class": SVC,
        "params": {"kernel": "linear", "C": 1.0}
    }
}

@ray.remote
def train_model(model_cls, X_train, y_train, params, model_name):
    """Función remota para entrenar un modelo específico"""
    model = model_cls(**params)
    model.fit(X_train, y_train)
    
    # Guardar modelo
    os.makedirs("../models", exist_ok=True)
    model_path = f"../models/{model_name}.joblib"
    dump(model, model_path)
    
    return {"model": model_name, "path": model_path, "status": "success"}

def load_dataset(path):
    """Carga y divide el dataset"""
    data = pd.read_csv(path)
    X = data.drop("target", axis=1)  # Asume que 'target' es la columna a predecir
    y = data["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def execute_distributed_training(dataset_path):
    """Orquesta el entrenamiento distribuido"""
    # 1. Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_dataset(dataset_path)
    
    # 2. Lanzar entrenamientos en paralelo
    futures = []
    for name, config in MODELS_CONFIG.items():
        future = train_model.remote(
            config["class"],
            X_train,
            y_train,
            config["params"],
            name
        )
        futures.append(future)
    
    # 3. Esperar y recoger resultados
    results = ray.get(futures)
    
    # 4. Verificar que los modelos se guardaron
    for result in results:
        if not os.path.exists(result["path"]):
            raise FileNotFoundError(f"Modelo {result['model']} no se guardó correctamente!")
    
    print("✅ Entrenamiento completado. Modelos guardados en /models/")
    return results
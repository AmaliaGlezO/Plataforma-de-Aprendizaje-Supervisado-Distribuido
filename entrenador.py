import ray
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import (
    LinearRegression, 
    Ridge
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(max_retries=3, retry_exceptions=True)
def train_model_remote(model, model_name, X_train, y_train, X_test, y_test, node_id=None):
    """Versión robusta con manejo de errores mejorado"""
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando {model_name} en nodo {node_id}")
        
        # Validación de datos de entrada
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("Datos contienen NaN después del preprocesamiento")
            
        # Pipeline completo con imputación
        pipeline = Pipeline([
            ('final_imputer', SimpleImputer(strategy='median')),  # Capa adicional de seguridad
            ('model', clone(model))
        ])
        
        # Entrenamiento
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} completado en {training_time:.2f}s")
        
        return {
            'model_name': model_name,
            'model': pipeline,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'training_time': training_time,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Fallo en {model_name}: {str(e)}", exc_info=True)
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e),
            'node_id': node_id
        }
    
class EntrenamientoDistribuido:
    def __init__(self, head_address=None, enable_fault_tolerance=True):
        """Inicializa el entrenador distribuido de ML con tolerancia a fallos"""
        self.enable_fault_tolerance = enable_fault_tolerance
        self.results = {}
        self.trained_models = {}
        self.failed_tasks = []
        self.cluster_nodes = []
        
        if not ray.is_initialized():
            ray_config = {
                "num_cpus": None,  
                "ignore_reinit_error": True,
            }
            
            if head_address:
                ray_config["address"] = head_address
                logger.info(f"Conectando a cluster Ray en: {head_address}")
            else:
                logger.info("Iniciando Ray en modo local con autodescubrimiento")
            
            ray.init(**ray_config)
        
        self._update_cluster_info()
        
    def _update_cluster_info(self):
        """Actualiza información del cluster para autodescubrimiento"""
        try:
            self.cluster_nodes = ray.nodes()
            alive_nodes = [node for node in self.cluster_nodes if node.get('Alive', False)]
            logger.info(f"Cluster autodescubierto: {len(alive_nodes)} nodos vivos de {len(self.cluster_nodes)} totales")
        except Exception as e:
            logger.warning(f"Error actualizando información del cluster: {e}") 

    def load_energy_data(self, filepath="data/datos_electricos.csv"):
        """Carga y prepara los datos con manejo robusto de NaN y sparse data"""
        logger.info(f"Cargando datos desde {filepath}")
        
        try:
            # Cargar datos
            df = pd.read_csv(filepath, parse_dates=['fecha'])
            
            # Análisis inicial de datos
            logger.info(f"Resumen de datos faltantes:\n{df.isnull().sum()}")
            logger.info(f"Total de valores NaN: {df.isnull().sum().sum()}")
            
            # Procesamiento de características
            df['year'] = df['fecha'].dt.year
            df['month'] = df['fecha'].dt.month
            df['day'] = df['fecha'].dt.day
            df['hour'] = df['fecha'].dt.hour
            df['day_of_week'] = df['fecha'].dt.dayofweek
            
            # Selección de características
            features = df.drop(columns=['fecha', 'deficit'])
            target = df['deficit'].copy()
            
            # Eliminar filas donde el target es NaN
            target = target.dropna()
            features = features.loc[target.index]
            
            # Pipeline de preprocesamiento robusto
            numeric_features = features.select_dtypes(include=np.number).columns
            categorical_features = features.select_dtypes(exclude=np.number).columns
            
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
            # Procesamiento final
            X = preprocessor.fit_transform(features)
            y = target.values
            
            # Convertir a dense array si es sparse
            if hasattr(X, 'toarray'):
                X = X.toarray()
                
            # Validación final
            if np.isnan(X).any():
                logger.warning("Aún hay NaN después del preprocesamiento - aplicando imputación adicional")
                X = SimpleImputer(strategy='median').fit_transform(X)
            
            logger.info(f"Datos finales - Forma: {X.shape}, NaN restantes: {np.isnan(X).sum()}")
            return X, y, features.columns.tolist()
            
        except Exception as e:
            logger.error(f"Error procesando datos: {str(e)}", exc_info=True)
            raise

    def get_available_models(self):
        """Retorna una lista de modelos de regresión robustos"""
        models = {
            'RandomForest': RandomForestRegressor(
                random_state=42, 
                n_estimators=50,
                max_depth=10,
                min_samples_split=5
            ),
            
            'HistGradientBoosting': HistGradientBoostingRegressor(
                random_state=42, 
                max_iter=50,
                learning_rate=0.1,
                max_depth=6
            ),
            
            'LinearRegression': LinearRegression(),
            
            'Ridge': Ridge(random_state=42, alpha=1.0),
            
            'GradientBoosting': GradientBoostingRegressor(
                random_state=42, 
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6
            )
        }
            
        return models 
           
    def train_models_distributed(self, selected_models=None, test_size=0.3):
        """Entrena múltiples modelos de regresión de forma distribuida"""
        logger.info("Iniciando entrenamiento distribuido para predicción de déficit de energía")
        
        self._update_cluster_info()

        # Cargar y preparar datos
        X, y, feature_names = self.load_energy_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        
        available_models = self.get_available_models()
        if selected_models is None:
            selected_models = list(available_models.keys())
        
        remote_tasks = []
        task_info = {}
        
        for i, model_name in enumerate(selected_models):
            if model_name in available_models:
                model = available_models[model_name]
                node_id = f"node_{i % max(len(self.cluster_nodes), 1)}"
                
                task = train_model_remote.remote(
                    model, model_name, X_train, y_train, X_test, y_test, node_id
                )
                remote_tasks.append(task)
                task_info[task] = {'model_name': model_name, 'node_id': node_id}
        
        logger.info(f"Ejecutando {len(remote_tasks)} entrenamientos en paralelo...")
        
        results = []
        failed_results = []
        
        try:
            completed_results = ray.get(remote_tasks, timeout=600)
            
            for result in completed_results:
                if result.get('status') == 'success':
                    results.append(result)
                else:
                    failed_results.append(result)
                    
        except ray.exceptions.GetTimeoutError:
            logger.warning("Timeout en algunas tareas, recuperando resultados parciales...")
            ready_tasks, _ = ray.wait(remote_tasks, num_returns=len(remote_tasks), timeout=0)
            
            for task in ready_tasks:
                try:
                    result = ray.get(task)
                    if result.get('status') == 'success':
                        results.append(result)
                    else:
                        failed_results.append(result)
                except Exception as e:
                    failed_results.append({
                        'model_name': task_info[task]['model_name'],
                        'status': 'failed',
                        'error': str(e),
                        'node_id': task_info[task]['node_id']
                    })
        
        for result in results:
            self.results[result['model_name']] = result
            self.trained_models[result['model_name']] = result['model']

        self.failed_tasks.extend(failed_results)

        total_tasks = len(selected_models)
        successful_tasks = len(results)
        failed_tasks = len(failed_results)
        
        logger.info(f"Resultados - Exitosos: {successful_tasks}/{total_tasks}, Fallos: {failed_tasks}")
        
        if results:
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['mse']
            )
            
            logger.info("\nResultados del entrenamiento:")
            logger.info("=" * 80)
            logger.info(f"{'Modelo':20} | {'MSE':<10} | {'MAE':<10} | {'R²':<10} | {'Tiempo':<10}")
            logger.info("=" * 80)
            for model_name, result in sorted_results:
                logger.info(f"{model_name:20} | {result['mse']:10.4f} | {result['mae']:10.4f} | "
                          f"{result['r2']:10.4f} | {result['training_time']:10.2f}s")
        
        return self.results
    
    def save_results(self, filename="training_results/training_results.json"):
        """Guarda los resultados en un archivo JSON"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Asegura que exista la carpeta
        with open(filename, 'w') as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in self.results.items()},
                f,
                indent=2
            )
        logger.info(f"Resultados guardados en: {filename}")

    def save_models(self, directory="models"):
        """Guarda los modelos entrenados"""
        os.makedirs(directory, exist_ok=True)
        saved_count = 0
        
        for model_name, model in self.trained_models.items():
            try:
                with open(f"{directory}/{model_name}.pkl", 'wb') as f:
                    pickle.dump(model, f)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error guardando {model_name}: {e}")
        
        logger.info(f"Modelos guardados: {saved_count}/{len(self.trained_models)}")
        return saved_count

def main():
    """Función principal para ejecutar el entrenamiento"""
    logger.info("Iniciando entrenador distribuido")

    trainer = EntrenamientoDistribuido()

    # Modelos seleccionados
    models_to_use = [
        'RandomForest', 
        'HistGradientBoosting',
        'LinearRegression', 
        'Ridge',
        'GradientBoosting'
    ]
    
    # Entrenar modelos
    results = trainer.train_models_distributed(selected_models=models_to_use)
    
    # Guardar resultados
    trainer.save_results()
    trainer.save_models()
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['mse'])
        logger.info(f"\n🏆 MEJOR MODELO: {best_model[0]}")
        logger.info(f"   - MSE: {best_model[1]['mse']:.4f}")
        logger.info(f"   - R²: {best_model[1]['r2']:.4f}")

    logger.info("\n✅ Entrenamiento completado!")

if __name__ == "__main__":
    main()
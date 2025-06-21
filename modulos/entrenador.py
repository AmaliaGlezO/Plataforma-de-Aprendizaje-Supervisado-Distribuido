import ray
import time
import json
import pandas as pd
from datetime import datetime
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import (
    LinearRegression, 
    SGDRegressor, 
    Ridge,
    Lasso,
    ElasticNet,
    PassiveAggressiveRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote(max_retries=3, retry_exceptions=True)
def train_model_remote(model, model_name, X_train, y_train, X_test, y_test, node_id=None):
    """Entrena un modelo de regresión de forma remota con tolerancia a fallos"""
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando entrenamiento de {model_name} en nodo {node_id}")
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Métricas de regresión
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_scores = -cv_scores  # Convertimos a positivo
        
        training_time = time.time() - start_time
        
        logger.info(f"Entrenamiento de {model_name} completado exitosamente en {training_time:.2f}s")
        
        return {
            'model_name': model_name,
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': y_pred.tolist(),
            'timestamp': datetime.now().isoformat(),
            'node_id': node_id,
            'status': 'success'
        }
    
    except Exception as e:
        logger.error(f"Error entrenando {model_name} en nodo {node_id}: {str(e)}")
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e),
            'node_id': node_id,
            'timestamp': datetime.now().isoformat()
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
                "_enable_object_reconstruction": True, 
                "_reconstruction_timeout": 30
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
        """Carga y prepara los datos de energía"""
        logger.info(f"Cargando datos de energía desde {filepath}")
        
        try:
            df = pd.read_csv(filepath, parse_dates=['fecha'])
            
            # Extraer características de fecha
            df['year'] = df['fecha'].dt.year
            df['month'] = df['fecha'].dt.month
            df['day'] = df['fecha'].dt.day
            df['hour'] = df['fecha'].dt.hour
            df['day_of_week'] = df['fecha'].dt.dayofweek
            
            # Definir características y objetivo
            features = df.drop(columns=['fecha', 'deficit', 'year', 'month'])  # Eliminamos las originales que hemos procesado
            target = df['deficit']
            
            # Preprocesamiento
            numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = ['horario_pico']  # Ejemplo de variable categórica
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(), categorical_features)
                ])
            
            return preprocessor.fit_transform(features), target, features.columns.tolist()
        
        except Exception as e:
            logger.error(f"Error cargando datos de energía: {e}")
            raise

    def get_available_models(self):
        """Retorna los modelos de regresión disponibles"""
        models = {
            # Modelos basados en árboles            
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
            'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
            'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=100, learning_rate=0.1),
            'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=100, max_depth=10),
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'XGBoost': HistGradientBoostingRegressor(random_state=42, max_iter=100, learning_rate=0.1, max_depth=6),
            
            # Modelos lineales
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42, alpha=1.0),
            'Lasso': Lasso(random_state=42, alpha=1.0),
            'ElasticNet': ElasticNet(random_state=42, alpha=1.0, l1_ratio=0.5),
            'SGD': SGDRegressor(random_state=42, max_iter=1000, alpha=0.0001),
            'PassiveAggressive': PassiveAggressiveRegressor(random_state=42, C=1.0, max_iter=1000),
            
            # Modelos basados en vecinos
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto'),
            
            # Support Vector Machines
            'SVR': SVR(C=1.0, kernel='rbf'),  
            'LinearSVR': LinearSVR(random_state=42, C=1.0, max_iter=1000),
            
            # Neural Networks
            'MLP': MLPRegressor(random_state=42, hidden_layer_sizes=(100,), max_iter=200, activation='relu', solver='adam'),
            
            # Ensemble Methods
            'Bagging': BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42), random_state=42, n_estimators=10),
            'Voting': VotingRegressor(estimators=[
                ('rf', RandomForestRegressor(random_state=42, n_estimators=50)),
                ('svr', SVR()),
                ('lr', LinearRegression())
            ])
        }
            
        return models 
           
    def train_models_distributed(self, selected_models=None, test_size=0.3):
        """Entrena múltiples modelos de regresión de forma distribuida con tolerancia a fallos"""
        logger.info("Iniciando entrenamiento distribuido para predicción de déficit de energía")
        
        self._update_cluster_info()

        # Cargar y preparar datos
        X, y, feature_names = self.load_energy_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Datos divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
        logger.info(f"Número de características: {X_train.shape[1]}")
        
        available_models = self.get_available_models()
        if selected_models is None:
            selected_models = list(available_models.keys())
        
        remote_tasks = []
        task_info = {}
        
        for i, model_name in enumerate(selected_models):
            if model_name in available_models:
                model = available_models[model_name]
                node_id = f"node_{i % len(self.cluster_nodes)}" if self.cluster_nodes else f"node_{i}"
                
                task = train_model_remote.remote(
                    model, model_name, X_train, y_train, X_test, y_test, node_id
                )
                remote_tasks.append(task)
                task_info[task] = {'model_name': model_name, 'node_id': node_id}
        
        logger.info(f"Ejecutando {len(remote_tasks)} entrenamientos en paralelo con tolerancia a fallos...")
        
        results = []
        failed_results = []
        
        try:
            completed_results = ray.get(remote_tasks, timeout=300)  
            
            for result in completed_results:
                if result.get('status') == 'success':
                    results.append(result)
                else:
                    failed_results.append(result)
                    
        except ray.exceptions.GetTimeoutError:
            logger.warning("Timeout en algunas tareas, recuperando resultados parciales...")
            ready_tasks, remaining_tasks = ray.wait(remote_tasks, num_returns=len(remote_tasks), timeout=0)
            
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
            
            for task in remaining_tasks:
                ray.cancel(task)
        
        except Exception as e:
            logger.error(f"Error durante ejecución distribuida: {e}")
            return {}
        
        for result in results:
            self.results[result['model_name']] = result
            self.trained_models[result['model_name']] = result['model']

        self.failed_tasks.extend(failed_results)

        total_tasks = len(selected_models)
        successful_tasks = len(results)
        failed_tasks = len(failed_results)
        
        logger.info(f"Tolerancia a fallos - Exitosos: {successful_tasks}/{total_tasks}, Fallos: {failed_tasks}")
        
        if failed_results:
            logger.warning("Tareas fallidas:")
            for failed in failed_results:
                logger.warning(f"  - {failed['model_name']}: {failed.get('error', 'Error desconocido')}")
        
        if results:
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['mse'], 
                reverse=False  # Ordenamos por MSE ascendente (menor MSE es mejor)
            )
            
            logger.info("\nResultados del entrenamiento distribuido:")
            logger.info("=" * 80)
            logger.info(f"{'Modelo':20} | {'MSE':<10} | {'MAE':<10} | {'R²':<10} | {'CV MSE':<15} | {'Tiempo':<10}")
            logger.info("=" * 80)
            for model_name, result in sorted_results:
                logger.info(f"{model_name:20} | {result['mse']:10.4f} | {result['mae']:10.4f} | "
                           f"{result['r2']:10.4f} | {result['cv_mean']:7.4f}±{result['cv_std']:5.4f} | "
                           f"{result['training_time']:10.2f}s")
        
        return self.results
    
    def save_results(self, filename="training_results.json"):
        """Guarda los resultados en un archivo JSON"""
        serializable_results = {}
        for model_name, result in self.results.items():
            serializable_result = result.copy()
            serializable_result.pop('model', None)
            serializable_results[model_name] = serializable_result        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Resultados guardados en: {filename}")
    
    def save_models(self, directory="models"):
        """Guarda TODOS los modelos entrenados exitosamente"""
        os.makedirs(directory, exist_ok=True)
        
        saved_count = 0
        for model_name, model in self.trained_models.items():
            try:
                filename = os.path.join(directory, f"{model_name}.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                saved_count += 1
                logger.info(f"Modelo {model_name} guardado en: {filename}")
            except Exception as e:
                logger.error(f"Error guardando modelo {model_name}: {e}")
        
        print(f"Modelos guardados en directorio: {directory} ({saved_count} modelos)")
        
        # También crear estructura para API
        api_directory = os.path.join("models", directory.split("_")[-1] if "_" in directory else "default")
        os.makedirs(api_directory, exist_ok=True)
        
        # Copiar modelos para la API
        import shutil
        for model_name, model in self.trained_models.items():
            try:
                source_file = os.path.join(directory, f"{model_name}.pkl")
                dest_file = os.path.join(api_directory, f"{model_name}.pkl")
                if os.path.exists(source_file):
                    shutil.copy2(source_file, dest_file)
                    logger.info(f"Modelo {model_name} copiado para API: {dest_file}")
            except Exception as e:
                logger.error(f"Error copiando modelo {model_name} para API: {e}")
    
    def get_cluster_info(self):
        """Obtiene información del cluster Ray"""
        return ray.cluster_resources()
    
    def get_fault_tolerance_stats(self):
        """Obtiene estadísticas de tolerancia a fallos"""
        return {
            'failed_tasks': len(self.failed_tasks),
            'failed_task_details': self.failed_tasks,
            'cluster_nodes': len(self.cluster_nodes),
            'alive_nodes': len([node for node in self.cluster_nodes if node.get('Alive', False)])
        }


def main():
    """Función principal para ejecutar el entrenamiento con tolerancia a fallos"""
    logger.info("Iniciando entrenador distribuido para predicción de déficit de energía")

    trainer = DistributedMLTrainer(enable_fault_tolerance=True)

    cluster_info = trainer.get_cluster_info()
    logger.info(f"Recursos del cluster autodescubierto: {cluster_info}")

    # Modelos seleccionados para evaluación
    models_to_use = [
        'RandomForest', 
        'GradientBoosting', 
        'LinearRegression', 
        'SVR', 
        'KNN',
        'XGBoost',
        'Ridge',
        'Lasso'
    ]
    
    # Entrenar modelos
    results = trainer.train_models_distributed(selected_models=models_to_use)
    
    # Guardar resultados
    trainer.save_results()
    trainer.save_models()
    
    # Mostrar estadísticas de fallos
    fault_stats = trainer.get_fault_tolerance_stats()
    logger.info(f"\n📊 ESTADÍSTICAS DE TOLERANCIA A FALLOS:")
    logger.info(f"Nodos en cluster: {fault_stats['cluster_nodes']}")
    logger.info(f"Nodos vivos: {fault_stats['alive_nodes']}")
    logger.info(f"Tareas fallidas: {fault_stats['failed_tasks']}")
    
    if fault_stats['failed_tasks'] > 0:
        logger.info("Detalles de tareas fallidas:")
        for failed_task in fault_stats['failed_task_details']:
            logger.info(f"  - {failed_task['model_name']}: {failed_task.get('error', 'Error desconocido')}")

    if results:
        best_model = min(results.items(), key=lambda x: x[1]['mse'])
        logger.info(f"\n🏆 MEJOR MODELO: {best_model[0]}")
        logger.info(f"   - MSE: {best_model[1]['mse']:.4f}")
        logger.info(f"   - MAE: {best_model[1]['mae']:.4f}")
        logger.info(f"   - R²: {best_model[1]['r2']:.4f}")
        logger.info(f"   - Tiempo de entrenamiento: {best_model[1]['training_time']:.2f}s")
    
    logger.info("\n✅ Entrenamiento distribuido completado con tolerancia a fallos!")
    logger.info(f"📁 Resultados guardados en training_results.json")
    logger.info(f"🤖 Modelos guardados en directorio models/")


if __name__ == "__main__":
    main()
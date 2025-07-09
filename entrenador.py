import ray
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
    VotingRegressor,
    StackingRegressor,
    # Clasificadores
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import (
    LinearRegression, 
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    TheilSenRegressor,
    RANSACRegressor,
    SGDRegressor,
    PassiveAggressiveRegressor,
    # Clasificadores
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    Perceptron
)
from sklearn.svm import SVR, LinearSVR, NuSVR, SVC, LinearSVC, NuSVC
from sklearn.neighbors import (
    KNeighborsRegressor, 
    RadiusNeighborsRegressor,
    KNeighborsClassifier,
    RadiusNeighborsClassifier
)
from sklearn.tree import (
    DecisionTreeRegressor, 
    ExtraTreeRegressor,
    DecisionTreeClassifier,
    ExtraTreeClassifier
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import pickle
import os
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Actor global para almacenar modelos y métricas
@ray.remote(name="ModeloStore", lifetime="detached", max_restarts=3, max_task_retries=5)
class ModeloStore:
    def __init__(self):
        self.modelos = {}  # model_name -> model
        self.metricas = {}  # model_name -> métricas
        self.datos_referencia = {}  # Para almacenar referencias de datos
        self.historial_entrenamientos = []  # Historial de entrenamientos
        
    def guardar_modelo(self, nombre, modelo, metricas):
        """Guarda un modelo entrenado y sus métricas"""
        self.modelos[nombre] = modelo
        self.metricas[nombre] = metricas
        
        # Agregar al historial
        self.historial_entrenamientos.append({
            'nombre': nombre,
            'timestamp': datetime.now().isoformat(),
            'metricas': metricas
        })
        
        return f"Modelo {nombre} guardado exitosamente"
    
    def obtener_modelo(self, nombre):
        """Obtiene un modelo específico"""
        return self.modelos.get(nombre)
    
    def obtener_metricas(self, nombre):
        """Obtiene métricas de un modelo específico"""
        return self.metricas.get(nombre)
    
    def listar_modelos(self):
        """Lista todos los modelos disponibles"""
        return list(self.modelos.keys())
    
    def listar_metricas(self):
        """Lista todas las métricas disponibles"""
        return self.metricas
    
    def obtener_estadisticas(self):
        """Obtiene estadísticas generales del almacén"""
        return {
            'total_modelos': len(self.modelos),
            'total_entrenamientos': len(self.historial_entrenamientos),
            'modelos_disponibles': list(self.modelos.keys())
        }
    
    def limpiar_modelos(self):
        """Limpia todos los modelos almacenados"""
        self.modelos.clear()
        self.metricas.clear()
        self.historial_entrenamientos.clear()
        return "Almacén de modelos limpiado"
    
    def guardar_datos_referencia(self, nombre, datos_ref):
        """Guarda referencias de datos en el object store"""
        self.datos_referencia[nombre] = datos_ref
        return f"Datos {nombre} guardados en object store"
    
    def obtener_datos_referencia(self, nombre):
        """Obtiene referencias de datos del object store"""
        return self.datos_referencia.get(nombre)
    

@ray.remote(max_retries=3, retry_exceptions=True)
def train_model_remote(model, model_name, X_train_ref, y_train_ref, X_test_ref, y_test_ref, task_type='regression', node_id=None):
    """Versión robusta con manejo de errores mejorado para regresión y clasificación usando referencias"""
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando {model_name} ({task_type}) en nodo {node_id}")
        
        # Obtener datos del object store
        X_train = ray.get(X_train_ref)
        y_train = ray.get(y_train_ref)
        X_test = ray.get(X_test_ref)
        y_test = ray.get(y_test_ref)
        
        # Validación de datos de entrada
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError("Datos contienen NaN después del preprocesamiento")
            
        # Pipeline completo con imputación
        pipeline = Pipeline([
            ('final_imputer', SimpleImputer(strategy='median')),
            ('model', clone(model))
        ])
        
        # Entrenamiento
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Métricas según el tipo de tarea
        if task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': float(mse),  # Convertir a float para serialización JSON
                'mae': float(mae),
                'r2': float(r2),
                'rmse': float(np.sqrt(mse))
            }
        else:  # classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        # Limpiar métricas de valores problemáticos (NaN, Inf)
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                metrics[key] = 0.0
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} completado en {training_time:.2f}s")
        
        result = {
            'model_name': model_name,
            'model': pipeline,
            'task_type': task_type,
            'training_time': float(training_time),
            'node_id': node_id,
            'status': 'success'
        }
        result.update(metrics)
        
        return result
        
    except Exception as e:
        logger.error(f"Fallo en {model_name}: {str(e)}", exc_info=True)
        return {
            'model_name': model_name,
            'task_type': task_type,
            'status': 'failed',
            'error': str(e),
            'node_id': node_id,
            'training_time': 0.0
        }

class EntrenamientoDistribuido:
    def __init__(self, head_address=None, enable_fault_tolerance=True):
        """Inicializa el entrenador distribuido de ML con tolerancia a fallos y actor global"""
        self.enable_fault_tolerance = enable_fault_tolerance
        self.results = {}
        self.trained_models = {}
        self.failed_tasks = []
        self.cluster_nodes = []
        
        # Inicializar Ray si no está inicializado
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
        
        # Inicializar o conectar al actor global ModeloStore
        self._init_modelo_store()
        self._update_cluster_info()
    
    def _init_modelo_store(self):
        """Inicializa o conecta al actor global ModeloStore"""
        try:
            # Intentar conectar al actor existente
            self.modelo_store = ray.get_actor("ModeloStore")
            logger.info("Actor global ModeloStore conectado exitosamente")
        except ValueError:
            # Si no existe, crear uno nuevo
            logger.info("Actor ModeloStore no existe. Creándolo nuevo...")
            self.modelo_store = ModeloStore.options(
                name="ModeloStore", 
                lifetime="detached", 
                get_if_exists=True
            ).remote()
            logger.info("Actor global ModeloStore creado exitosamente")
        
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
            df['month_num'] = df['fecha'].dt.month
            df['day'] = df['fecha'].dt.day
            df['hour'] = df['fecha'].dt.hour
            df['day_of_week'] = df['fecha'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Características adicionales del dataset energético
            if 'disponibilidad' in df.columns and 'demanda_maxima' in df.columns:
                df['capacidad_utilizada'] = df['disponibilidad'] / df['demanda_maxima']
                df['deficit_ratio'] = df['deficit'] / df['demanda_maxima']
                df['respaldo_ratio'] = df['respaldo'] / df['disponibilidad']
            
            # Selección de características
            features = df.drop(columns=['fecha', 'deficit'] + (['month'] if 'month' in df.columns else []))
            target = df['deficit'].copy()
            
            # Eliminar filas donde el target es NaN
            valid_idx = ~target.isna()
            target = target[valid_idx]
            features = features.loc[valid_idx]
            
            # Pipeline de preprocesamiento robusto
            numeric_features = features.select_dtypes(include=np.number).columns
            categorical_features = features.select_dtypes(exclude=np.number).columns
            
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
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
            
            # Crear target de clasificación (alto/bajo déficit)
            y_classification = (y > np.median(y)).astype(int)
            
            return X, y, y_classification, features.columns.tolist()
            
        except Exception as e:
            logger.error(f"Error procesando datos: {str(e)}", exc_info=True)
            raise

    def get_regression_models(self):
        """Retorna modelos de regresión expandidos"""
        models = {
            # Ensemble Methods
            'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10),
            'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=100, max_depth=10),
            'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42, max_iter=100),
            'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=50),
            'Bagging': BaggingRegressor(random_state=42, n_estimators=50),
            
            # Linear Models
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42, alpha=1.0),
            'Lasso': Lasso(random_state=42, alpha=1.0),
            'ElasticNet': ElasticNet(random_state=42, alpha=1.0),
            'BayesianRidge': BayesianRidge(),
            'HuberRegressor': HuberRegressor(),
            'TheilSenRegressor': TheilSenRegressor(random_state=42),
            'RANSACRegressor': RANSACRegressor(random_state=42),
            'SGDRegressor': SGDRegressor(random_state=42),
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=42),
            
            # Support Vector Machines
            'SVR': SVR(kernel='rbf', C=1.0),
            'LinearSVR': LinearSVR(random_state=42, max_iter=2000),
            'NuSVR': NuSVR(kernel='rbf', C=1.0),
            
            # Tree-based
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'ExtraTree': ExtraTreeRegressor(random_state=42, max_depth=10),
            
            # Neighbors
            'KNeighbors': KNeighborsRegressor(n_neighbors=5),
            'RadiusNeighbors': RadiusNeighborsRegressor(radius=1.0),
            
            # Neural Networks
            'MLPRegressor': MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(100,)),
            
            # Gaussian Process
            'GaussianProcess': GaussianProcessRegressor(random_state=42),
            
            # Other
            'KernelRidge': KernelRidge(alpha=1.0),
            'PLSRegression': PLSRegression(n_components=2),
            'DummyRegressor': DummyRegressor(strategy='mean'),
        }
        
        # Ensemble methods
        base_models = [
            ('rf', RandomForestRegressor(random_state=42, n_estimators=10)),
            ('gb', GradientBoostingRegressor(random_state=42, n_estimators=10)),
            ('lr', LinearRegression())
        ]
        
        models['VotingRegressor'] = VotingRegressor(estimators=base_models)
        models['StackingRegressor'] = StackingRegressor(
            estimators=base_models,
            final_estimator=LinearRegression(),
            cv=3
        )
        
        return models

    def get_classification_models(self):
        """Retorna modelos de clasificación expandidos"""
        models = {
            # Ensemble Methods
            'RandomForestClassifier': RandomForestClassifier(random_state=42, n_estimators=100),
            'ExtraTreesClassifier': ExtraTreesClassifier(random_state=42, n_estimators=100),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42, max_iter=100),
            'AdaBoostClassifier': AdaBoostClassifier(random_state=42, n_estimators=50),
            'BaggingClassifier': BaggingClassifier(random_state=42, n_estimators=50),
            
            # Linear Models
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SGDClassifier': SGDClassifier(random_state=42, max_iter=1000),
            'RidgeClassifier': RidgeClassifier(random_state=42),
            'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=42),
            'Perceptron': Perceptron(random_state=42),
            
            # Support Vector Machines
            'SVC': SVC(random_state=42, kernel='rbf'),
            'LinearSVC': LinearSVC(random_state=42, max_iter=2000),
            'NuSVC': NuSVC(random_state=42, kernel='rbf'),
            
            # Tree-based
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42, max_depth=10),
            'ExtraTreeClassifier': ExtraTreeClassifier(random_state=42, max_depth=10),
            
            # Neighbors
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
            
            # Discriminant Analysis
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
            
            # Gaussian Process
            'GaussianProcessClassifier': GaussianProcessClassifier(random_state=42),
            
            # Other
            'DummyClassifier': DummyClassifier(strategy='most_frequent', random_state=42),
        }
        
        # Ensemble methods
        base_models = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=10)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=10)),
            ('lr', LogisticRegression(random_state=42, max_iter=500))
        ]
        
        models['VotingClassifier'] = VotingClassifier(estimators=base_models, voting='soft')
        models['StackingClassifier'] = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=3
        )
        
        return models
           
    def train_models_distributed(self, task_type='both', selected_models=None, test_size=0.3):
        """Entrena múltiples modelos de forma distribuida usando Ray actors"""
        logger.info(f"Iniciando entrenamiento distribuido - Tipo de tarea: {task_type}")
        
        self._update_cluster_info()

        # Cargar y preparar datos
        X, y_reg, y_clf, feature_names = self.load_energy_data()
        
        remote_tasks = []
        task_info = {}
        
        # Entrenar modelos de regresión
        if task_type in ['regression', 'both']:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_reg, test_size=test_size, random_state=42
            )
            
            logger.info(f"Datos de regresión divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
            
            # Poner los datos en el object store de Ray
            X_train_ref = ray.put(X_train)
            X_test_ref = ray.put(X_test)
            y_train_ref = ray.put(y_train)
            y_test_ref = ray.put(y_test)
            
            # Guardar referencias en el actor para uso posterior
            ray.get(self.modelo_store.guardar_datos_referencia.remote(
                "regression_data", {
                    'X_train': X_train_ref,
                    'X_test': X_test_ref,
                    'y_train': y_train_ref,
                    'y_test': y_test_ref
                }
            ))
            
            regression_models = self.get_regression_models()
            models_to_use = selected_models if selected_models else list(regression_models.keys())
            
            for i, model_name in enumerate(models_to_use):
                if model_name in regression_models:
                    model = regression_models[model_name]
                    node_id = f"node_{i % max(len(self.cluster_nodes), 1)}"
                    
                    task = train_model_remote.remote(
                        model, f"{model_name}_REG", X_train_ref, y_train_ref, X_test_ref, y_test_ref, 'regression', node_id
                    )
                    remote_tasks.append(task)
                    task_info[task] = {'model_name': f"{model_name}_REG", 'node_id': node_id}
        
        # Entrenar modelos de clasificación
        if task_type in ['classification', 'both']:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_clf, test_size=test_size, random_state=42
            )
            
            logger.info(f"Datos de clasificación divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
            
            # Poner los datos en el object store de Ray
            X_train_ref = ray.put(X_train)
            X_test_ref = ray.put(X_test)
            y_train_ref = ray.put(y_train)
            y_test_ref = ray.put(y_test)
            
            # Guardar referencias en el actor para uso posterior
            ray.get(self.modelo_store.guardar_datos_referencia.remote(
                "classification_data", {
                    'X_train': X_train_ref,
                    'X_test': X_test_ref,
                    'y_train': y_train_ref,
                    'y_test': y_test_ref
                }
            ))
            
            classification_models = self.get_classification_models()
            models_to_use = selected_models if selected_models else list(classification_models.keys())
            
            for i, model_name in enumerate(models_to_use):
                if model_name in classification_models:
                    model = classification_models[model_name]
                    node_id = f"node_{i % max(len(self.cluster_nodes), 1)}"
                    
                    task = train_model_remote.remote(
                        model, f"{model_name}_CLF", X_train_ref, y_train_ref, X_test_ref, y_test_ref, 'classification', node_id
                    )
                    remote_tasks.append(task)
                    task_info[task] = {'model_name': f"{model_name}_CLF", 'node_id': node_id}
        
        logger.info(f"Ejecutando {len(remote_tasks)} entrenamientos en paralelo...")
        
        results = []
        failed_results = []
        
        try:
            completed_results = ray.get(remote_tasks, timeout=1200)  # 20 minutos timeout
            
            for result in completed_results:
                if result.get('status') == 'success':
                    results.append(result)
                    
                    # Guardar en el actor global
                    metricas_limpias = {
                        k: v for k, v in result.items()
                        if k in ['mse', 'mae', 'r2', 'rmse', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'task_type']
                    }
                    
                    ray.get(self.modelo_store.guardar_modelo.remote(
                        result['model_name'], result['model'], metricas_limpias
                    ))
                    
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
                        
                        # Guardar en el actor global
                        metricas_limpias = {
                            k: v for k, v in result.items()
                            if k in ['mse', 'mae', 'r2', 'rmse', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'task_type']
                        }
                        
                        ray.get(self.modelo_store.guardar_modelo.remote(
                            result['model_name'], result['model'], metricas_limpias
                        ))
                        
                    else:
                        failed_results.append(result)
                except Exception as e:
                    failed_results.append({
                        'model_name': task_info[task]['model_name'],
                        'status': 'failed',
                        'error': str(e),
                        'node_id': task_info[task]['node_id']
                    })
        
        # Almacenar resultados localmente también
        for result in results:
            self.results[result['model_name']] = result
            self.trained_models[result['model_name']] = result['model']

        self.failed_tasks.extend(failed_results)

        # Mostrar resultados
        self._display_results()
        
        return self.results
    
    def _display_results(self):
        """Muestra los resultados del entrenamiento"""
        total_tasks = len(self.results) + len(self.failed_tasks)
        successful_tasks = len(self.results)
        failed_tasks = len(self.failed_tasks)

        print(f"Resultados - Exitosos: {successful_tasks}/{total_tasks}, Fallos: {failed_tasks}")

        if self.results:
            regression_results = {k: v for k, v in self.results.items() if v.get('task_type') == 'regression'}
            classification_results = {k: v for k, v in self.results.items() if v.get('task_type') == 'classification'}

            if regression_results:
                print("\n" + "="*80)
                print("RESULTADOS DE REGRESIÓN")
                print("="*80)
                sorted_reg = sorted(regression_results.items(), key=lambda x: x[1]['mse'])
                print(f"{'Modelo':30} | {'MSE':<10} | {'MAE':<10} | {'R²':<10} | {'RMSE':<10} | {'Tiempo':<8}")
                print("-"*80)
                for model_name, result in sorted_reg:
                    print(f"{model_name:30} | {result['mse']:<10.4f} | {result['mae']:<10.4f} | {result['r2']:<10.4f} | {result['rmse']:<10.4f} | {result['training_time']:<8.2f}s")

            if classification_results:
                print("\n" + "="*80)
                print("RESULTADOS DE CLASIFICACIÓN")
                print("="*80)
                sorted_clf = sorted(classification_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                print(f"{'Modelo':30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Tiempo':<8}")
                print("-"*80)
                for model_name, result in sorted_clf:
                    print(f"{model_name:30} | {result['accuracy']:<10.4f} | {result['precision']:<10.4f} | {result['recall']:<10.4f} | {result['f1_score']:<10.4f} | {result['training_time']:<8.2f}s")

    def save_results_json(self, output_path="resultados_entrenamiento.json"):
        import json
        with open(output_path, 'w') as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in self.results.items()}, f, indent=4)
        print(f"Resultados guardados en {output_path}")

    def save_models_pickle(self, output_dir="modelos_guardados"):
        import os
        os.makedirs(output_dir, exist_ok=True)
        for model_name, model in self.trained_models.items():
            filepath = os.path.join(output_dir, f"{model_name}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        print(f"Modelos guardados en {output_dir}")

    def listar_modelos_guardados(self):
        return ray.get(self.modelo_store.listar_modelos.remote())

    def obtener_metricas_modelo(self, nombre_modelo):
        return ray.get(self.modelo_store.obtener_metricas.remote(nombre_modelo))

    def limpiar_almacen_modelos(self):
        return ray.get(self.modelo_store.limpiar_modelos.remote())

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



@ray.remote(max_retries=3, retry_exceptions=True)
def train_model_remote(model, model_name, X_train, y_train, X_test, y_test, task_type='regression', node_id=None):
    """Versión robusta con manejo de errores mejorado para regresión y clasificación"""
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando {model_name} ({task_type}) en nodo {node_id}")
        
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
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
        else:  # classification
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} completado en {training_time:.2f}s")
        
        result = {
            'model_name': model_name,
            'model': pipeline,
            'task_type': task_type,
            'training_time': training_time,
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
        """Entrena múltiples modelos de forma distribuida"""
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
            
            regression_models = self.get_regression_models()
            models_to_use = selected_models if selected_models else list(regression_models.keys())
            
            for i, model_name in enumerate(models_to_use):
                if model_name in regression_models:
                    model = regression_models[model_name]
                    node_id = f"node_{i % max(len(self.cluster_nodes), 1)}"
                    
                    task = train_model_remote.remote(
                        model, f"{model_name}_REG", X_train, y_train, X_test, y_test, 'regression', node_id
                    )
                    remote_tasks.append(task)
                    task_info[task] = {'model_name': f"{model_name}_REG", 'node_id': node_id}
        
        # Entrenar modelos de clasificación
        if task_type in ['classification', 'both']:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_clf, test_size=test_size, random_state=42
            )
            
            logger.info(f"Datos de clasificación divididos: {X_train.shape[0]} entrenamiento, {X_test.shape[0]} prueba")
            
            classification_models = self.get_classification_models()
            models_to_use = selected_models if selected_models else list(classification_models.keys())
            
            for i, model_name in enumerate(models_to_use):
                if model_name in classification_models:
                    model = classification_models[model_name]
                    node_id = f"node_{i % max(len(self.cluster_nodes), 1)}"
                    
                    task = train_model_remote.remote(
                        model, f"{model_name}_CLF", X_train, y_train, X_test, y_test, 'classification', node_id
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
        
        # Almacenar resultados
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
        
        logger.info(f"Resultados - Exitosos: {successful_tasks}/{total_tasks}, Fallos: {failed_tasks}")
        
        if self.results:
            # Separar por tipo de tarea
            regression_results = {k: v for k, v in self.results.items() if v.get('task_type') == 'regression'}
            classification_results = {k: v for k, v in self.results.items() if v.get('task_type') == 'classification'}
            
            # Mostrar resultados de regresión
            if regression_results:
                logger.info("\n" + "="*100)
                logger.info("RESULTADOS DE REGRESIÓN")
                logger.info("="*100)
                sorted_reg = sorted(regression_results.items(), key=lambda x: x[1]['mse'])
                logger.info(f"{'Modelo':30} | {'MSE':<12} | {'MAE':<12} | {'R²':<12} | {'RMSE':<12} | {'Tiempo':<10}")
                logger.info("-"*100)
                for model_name, result in sorted_reg:
                    logger.info(f"{model_name:30} | {result['mse']:12.4f} | {result['mae']:12.4f} | "
                              f"{result['r2']:12.4f} | {result['rmse']:12.4f} | {result['training_time']:10.2f}s")
            
            # Mostrar resultados de clasificación
            if classification_results:
                logger.info("\n" + "="*100)
                logger.info("RESULTADOS DE CLASIFICACIÓN")
                logger.info("="*100)
                sorted_clf = sorted(classification_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                logger.info(f"{'Modelo':30} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Tiempo':<10}")
                logger.info("-"*100)
                for model_name, result in sorted_clf:
                    logger.info(f"{model_name:30} | {result['accuracy']:12.4f} | {result['precision']:12.4f} | "
                              f"{result['recall']:12.4f} | {result['f1_score']:12.4f} | {result['training_time']:10.2f}s")
    
    def save_results(self, filename="training_results/expanded_training_results.json"):
        """Guarda los resultados en un archivo JSON"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in self.results.items()},
                f,
                indent=2
            )
        logger.info(f"Resultados guardados en: {filename}")

    def save_models(self, directory="models_expanded"):
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

    def print_cluster_info(self):
        """Imprime información sobre los nodos del clúster"""
        self._update_cluster_info()
        logger.info(f"Nodos en el clúster: {len(self.cluster_nodes)}")
        for node in self.cluster_nodes:
            logger.info(f"ID: {node['NodeID']}, Alive: {node['Alive']}, IP: {node['NodeManagerAddress']}")

def main():
    """Función principal para ejecutar el entrenamiento expandido"""
    logger.info("Iniciando entrenador distribuido expandido")

    trainer = EntrenamientoDistribuido()
    
    # Entrenar todos los modelos (regresión y clasificación)
    results = trainer.train_models_distributed(task_type='both')
    
    # Guardar resultados
    trainer.save_results()
    trainer.save_models()
    
    if results:
        # Mejores modelos por categoría
        regression_results = {k: v for k, v in results.items() if v.get('task_type') == 'regression'}
        classification_results = {k: v for k, v in results.items() if v.get('task_type') == 'classification'}
        
        if regression_results:
            best_reg = min(regression_results.items(), key=lambda x: x[1]['mse'])
            logger.info(f"\n🏆 MEJOR MODELO DE REGRESIÓN: {best_reg[0]}")
            logger.info(f"   - MSE: {best_reg[1]['mse']:.4f}")
            logger.info(f"   - R²: {best_reg[1]['r2']:.4f}")
        
        if classification_results:
            best_clf = max(classification_results.items(), key=lambda x: x[1]['accuracy'])
            logger.info(f"\n🏆 MEJOR MODELO DE CLASIFICACIÓN: {best_clf[0]}")
            logger.info(f"   - Accuracy: {best_clf[1]['accuracy']:.4f}")
            logger.info(f"   - F1-Score: {best_clf[1]['f1_score']:.4f}")

    logger.info(f"\n✅ Entrenamiento expandido completado!")
    logger.info(f"Total de modelos entrenados: {len(results)}")

if __name__ == "__main__":
    main()
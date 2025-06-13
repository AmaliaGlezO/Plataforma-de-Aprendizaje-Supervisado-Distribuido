"""
CARGADOR DE DATOS PARA DATASET DE CONSUMO ENERGÉTICO
Maneja la carga, procesamiento y particionamiento del dataset energy_consumption
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple, List, Dict, Any
import json
from pathlib import Path

class DataLoader:
    """Maneja carga y procesamiento de datasets de consumo energético"""
    
    def __init__(self):
        """Inicializa el cargador de datos con configuraciones específicas"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.preprocessor = None
        self.feature_columns = [
            'Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
            'HVACUsage', 'LightingUsage', 'RenewableEnergy',
            'DayOfWeek', 'Holiday'
        ]
        self.target_column = 'EnergyConsumption'
        self.categorical_cols = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']
        self.numerical_cols = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']
    
    def load_dataset(self, file_path: str = "data/datasets/energy_consumption.csv") -> pd.DataFrame:
        """
        Carga el dataset de consumo energético desde un archivo CSV
        Args:
            file_path: Ruta al archivo CSV
        Returns:
            DataFrame con los datos cargados
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en {file_path}")
        
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        df.sort_values('Timestamp', inplace=True)
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida la calidad y estructura del dataset
        Args:
            df: DataFrame a validar
        Returns:
            Diccionario con resultados de validación
        """
        validation = {
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'rows': len(df),
            'columns': list(df.columns),
            'duplicates': df.duplicated().sum()
        }
        return validation
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa los datos: escalado, encoding, etc.
        Args:
            df: DataFrame con datos crudos
        Returns:
            Tupla con (X, y) preprocesados
        """
        # Definir el preprocesador
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(), self.categorical_cols)
            ],
            remainder='passthrough'
        )
        
        X = df[self.feature_columns]
        y = df[self.target_column].values
        
        # Aplicar transformaciones
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, y
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """
        Divide datos en conjuntos de entrenamiento y prueba
        Args:
            X: Features
            y: Target
            test_size: Proporción para test
        Returns:
            Tupla con (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def create_data_partitions(self, X: np.ndarray, y: np.ndarray, num_partitions: int) -> List[Tuple]:
        """
        Crea particiones de datos para distribución
        Args:
            X: Features
            y: Target
            num_partitions: Número de particiones a crear
        Returns:
            Lista de tuplas (X_part, y_part) para cada partición
        """
        # Mejor estrategia para datos temporales: particiones temporales
        partition_size = len(X) // num_partitions
        partitions = []
        
        for i in range(num_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i < num_partitions - 1 else len(X)
            
            X_part = X[start:end]
            y_part = y[start:end]
            partitions.append((X_part, y_part))
        
        return partitions
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maneja valores faltantes en el dataset
        Args:
            df: DataFrame con posibles valores faltantes
        Returns:
            DataFrame limpio
        """
        # Para datos temporales, interpolación es mejor que eliminación
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
        
        # Para categóricas, usamos moda
        for col in self.categorical_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calcula estadísticas descriptivas del dataset
        Args:
            df: DataFrame a analizar
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'summary': df.describe().to_dict(),
            'correlation_with_target': df.corr()[self.target_column].to_dict(),
            'temporal_coverage': {
                'start': df['Timestamp'].min().isoformat(),
                'end': df['Timestamp'].max().isoformat()
            },
            'energy_consumption_stats': {
                'mean': df[self.target_column].mean(),
                'std': df[self.target_column].std(),
                'min': df[self.target_column].min(),
                'max': df[self.target_column].max()
            }
        }
        return stats
    
    def save_preprocessor(self, path: str = "data/models/preprocessor.pkl"):
        """
        Guarda el preprocesador para uso futuro
        Args:
            path: Ruta donde guardar el preprocesador
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor no ha sido entrenado aún")
        
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, path)
    
    def load_preprocessor(self, path: str = "data/models/preprocessor.pkl"):
        """
        Carga un preprocesador guardado
        Args:
            path: Ruta al preprocesador guardado
        """
        import joblib
        self.preprocessor = joblib.load(path)
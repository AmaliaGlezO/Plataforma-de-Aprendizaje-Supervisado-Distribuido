import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def validate_input_data(data):
    """Valida datos de entrada (JSON desde frontend)"""
    pass

def clean_data(df):
    """Limpia datos: maneja NaN, inf, valores atípicos"""
    pass

def preprocess_data(df, target_column):
    """Preprocesa datos para entrenamiento"""
    pass

def split_data(X, y, test_size=0.2, random_state=42):
    """Divide datos en train/test"""
    pass

def encode_categorical_variables(df):
    """Codifica variables categóricas"""
    pass

def scale_numerical_features(X_train, X_test):
    """Escala características numéricas"""
    pass

def prepare_data_for_training(raw_data, target_column):
    """Pipeline completo de preparación de datos"""
    pass
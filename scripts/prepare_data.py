#!/usr/bin/env python3
"""
Script para preparar y generar datos de energía para entrenamiento
Basado en la muestra proporcionada del dataset energético
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_energy_data(num_samples=1000, output_path='data/energia_dataset.csv'):
    """
    Genera datos sintéticos basados en la muestra del dataset energético
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # Fechas base
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Mapeo de meses
    month_names = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                   'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    
    data = []
    
    for i, date in enumerate(dates):
        # Generar datos realistas basados en patrones energéticos
        
        # Disponibilidad (2000-3000 MW típicamente)
        disponibilidad = np.random.normal(2500, 300)
        disponibilidad = max(1800, min(3200, disponibilidad))
        
        # Demanda máxima (generalmente mayor que disponibilidad)
        demanda_base = disponibilidad + np.random.normal(400, 200)
        demanda_maxima = max(disponibilidad + 50, demanda_base)
        
        # Afectación (diferencia entre demanda y disponibilidad más factores)
        afectacion_base = demanda_maxima - disponibilidad
        afectacion = max(0, afectacion_base + np.random.normal(0, 100))
        
        # Déficit (relacionado con afectación pero puede ser menor)
        deficit = max(0, afectacion - np.random.exponential(100))
        
        # Respaldo (0 o 1, más probable cuando hay déficit alto)
        respaldo = 1 if deficit > 300 and np.random.random() < 0.7 else 0
        
        # Horario pico (0-24, más común en ciertas horas)
        if np.random.random() < 0.3:  # 30% de probabilidad de horario pico
            horario_pico = np.random.choice([18, 19, 20, 21])  # Horas pico típicas
        else:
            horario_pico = np.random.randint(0, 24)
        
        # Unidades en avería (0-10)
        unidades_averia = np.random.poisson(3)
        unidades_averia = min(10, unidades_averia)
        
        # Unidades en mantenimiento (0-5)
        unidades_mantenimiento = np.random.poisson(1)
        unidades_mantenimiento = min(5, unidades_mantenimiento)
        
        # Limitación térmica (relacionada con temperatura/época del año)
        season_factor = np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        limitacion_termica = max(0, 300 + season_factor * 200 + np.random.normal(0, 50))
        
        # Motores impacto (relacionado con disponibilidad)
        motores_impacto = disponibilidad * 0.35 + np.random.normal(0, 50)
        motores_impacto = max(500, min(1000, motores_impacto))
        
        # Crear registro
        record = {
            'fecha': date.strftime('%Y-%m-%d %H:%M:%S'),
            'year': date.year,
            'month': month_names[date.month - 1],
            'disponibilidad': round(disponibilidad, 1),
            'demanda_maxima': round(demanda_maxima, 1),
            'afectacion': round(afectacion, 1),
            'deficit': round(deficit, 1),
            'respaldo': respaldo,
            'horario_pico': horario_pico,
            'unidades_averia': unidades_averia,
            'unidades_mantenimiento': unidades_mantenimiento,
            'limitacion_termica': round(limitacion_termica, 1),
            'motores_impacto': round(motores_impacto, 1)
        }
        
        data.append(record)
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset sintético generado: {output_path}")
    logger.info(f"Forma del dataset: {df.shape}")
    logger.info(f"Estadísticas del déficit:")
    logger.info(df['deficit'].describe())
    
    return df

def load_sample_data():
    """
    Carga los datos de muestra proporcionados
    """
    sample_data = [
        {
            'fecha': '2022-12-02 10:34:00',
            'year': 2022,
            'month': 'diciembre',
            'disponibilidad': 2235.0,
            'demanda_maxima': 3170.0,
            'afectacion': 1005.0,
            'deficit': 935.0,
            'respaldo': 0,
            'horario_pico': 0,
            'unidades_averia': 4,
            'unidades_mantenimiento': 0,
            'limitacion_termica': 313.0,
            'motores_impacto': 918.0
        },
        {
            'fecha': '2022-12-03 10:41:00',
            'year': 2022,
            'month': 'diciembre',
            'disponibilidad': 2419.0,
            'demanda_maxima': 3080.0,
            'afectacion': 731.0,
            'deficit': 661.0,
            'respaldo': 0,
            'horario_pico': 82,
            'unidades_averia': 5,
            'unidades_mantenimiento': 2,
            'limitacion_termica': 271.0,
            'motores_impacto': 915.0
        },
        # ... resto de datos de muestra
    ]
    
    return pd.DataFrame(sample_data)

def analyze_data(df):
    """
    Analiza el dataset generado
    """
    logger.info("\n📊 ANÁLISIS DEL DATASET")
    logger.info("="*40)
    logger.info(f"Número total de registros: {len(df)}")
    logger.info(f"Rango de fechas: {df['fecha'].min()} - {df['fecha'].max()}")
    
    # Estadísticas de la variable objetivo
    logger.info(f"\n🎯 VARIABLE OBJETIVO (déficit):")
    logger.info(f"Media: {df['deficit'].mean():.2f}")
    logger.info(f"Mediana: {df['deficit'].median():.2f}")
    logger.info(f"Desviación estándar: {df['deficit'].std():.2f}")
    logger.info(f"Mínimo: {df['deficit'].min():.2f}")
    logger.info(f"Máximo: {df['deficit'].max():.2f}")
    
    # Correlaciones importantes
    logger.info(f"\n🔗 CORRELACIONES CON DÉFICIT:")
    correlations = df.select_dtypes(include=[np.number]).corr()['deficit'].sort_values(ascending=False)
    for var, corr in correlations.items():
        if var != 'deficit' and abs(corr) > 0.1:
            logger.info(f"{var}: {corr:.3f}")

def main():
    """Función principal"""
    logger.info("📊 PREPARACIÓN DE DATOS ENERGÉTICOS")
    logger.info("="*40)
    
    # Generar dataset sintético
    df = generate_synthetic_energy_data(num_samples=2000)
    
    # Analizar datos
    analyze_data(df)
    
    logger.info("\n✅ Datos preparados correctamente!")
    logger.info("Ahora puedes ejecutar el entrenamiento distribuido con:")
    logger.info("python scripts/train_and_deploy.py")

if __name__ == "__main__":
    main()
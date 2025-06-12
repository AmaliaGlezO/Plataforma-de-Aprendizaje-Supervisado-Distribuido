#!/usr/bin/env python3
"""
Script principal para entrenamiento distribuido de modelos de ML
para predicción de déficit energético usando Ray
"""

import ray
import os
import sys
import pandas as pd
import json
from datetime import datetime
import logging

# Agregar el directorio raíz al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.distributed_trainer import train_model, get_model_configurations

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Crear directorios necesarios"""
    directories = ['models/', 'logs/', 'results/']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {directory}")

def validate_dataset(data_path):
    """Validar que el dataset existe y tiene la estructura correcta"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset no encontrado en: {data_path}")
    
    # Cargar y validar estructura
    df = pd.read_csv(data_path)
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    required_columns = [
        'fecha', 'year', 'month', 'disponibilidad', 'demanda_maxima',
        'afectacion', 'deficit', 'respaldo', 'horario_pico',
        'unidades_averia', 'unidades_mantenimiento', 'limitacion_termica',
        'motores_impacto'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columnas faltantes en el dataset: {missing_columns}")
    
    logger.info("✓ Dataset validado correctamente")
    return df

def initialize_ray():
    """Inicializar Ray con configuración optimizada"""
    if ray.is_initialized():
        ray.shutdown()
    
    # Configuración de Ray
    ray.init(
        num_cpus=None,  # Usar todos los CPUs disponibles
        ignore_reinit_error=True,
        logging_level=logging.INFO
    )
    
    logger.info("✓ Ray inicializado")
    logger.info(f"Recursos disponibles: {ray.available_resources()}")

def run_distributed_training(data_path, output_dir='models/'):
    """
    Ejecutar entrenamiento distribuido de múltiples modelos
    """
    logger.info("🚀 Iniciando entrenamiento distribuido...")
    
    # Obtener configuraciones de modelos
    configurations = get_model_configurations()
    logger.info(f"Se entrenarán {len(configurations)} modelos en paralelo")
    
    # Lanzar entrenamientos en paralelo usando Ray
    training_futures = []
    
    for config in configurations:
        future = train_model.remote(
            data_path=data_path,
            model_type=config['model_type'],
            hyperparams=config['hyperparams'],
            model_name=config['model_name'],
            output_dir=output_dir
        )
        training_futures.append(future)
        logger.info(f"Lanzado entrenamiento: {config['model_name']} ({config['model_type']})")
    
    # Esperar a que todos los entrenamientos terminen
    logger.info("⏳ Esperando resultados de entrenamientos paralelos...")
    results = ray.get(training_futures)
    
    return results

def analyze_results(results):
    """Analizar y mostrar resultados de entrenamientos"""
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMEN DE ENTRENAMIENTOS")
    logger.info("="*60)
    
    successful_models = []
    failed_models = []
    
    for result in results:
        if result['status'] == 'success':
            successful_models.append(result)
            metrics = result['metrics']
            logger.info(f"\n✅ {result['model_name']} ({result['model_type']})")
            logger.info(f"   Test R²: {metrics['test_r2']:.4f}")
            logger.info(f"   Test MAE: {metrics['test_mae']:.2f}")
            logger.info(f"   Test MSE: {metrics['test_mse']:.2f}")
            logger.info(f"   Tiempo: {result['training_time']:.2f}s")
        else:
            failed_models.append(result)
            logger.error(f"\n❌ {result['model_name']}: {result.get('error', 'Error desconocido')}")
    
    # Encontrar mejor modelo
    if successful_models:
        best_model = max(successful_models, key=lambda x: x['metrics']['test_r2'])
        logger.info(f"\n🏆 MEJOR MODELO: {best_model['model_name']}")
        logger.info(f"   Test R²: {best_model['metrics']['test_r2']:.4f}")
        logger.info(f"   Test MAE: {best_model['metrics']['test_mae']:.2f}")
    
    logger.info(f"\n📈 RESUMEN FINAL:")
    logger.info(f"   Modelos exitosos: {len(successful_models)}/{len(results)}")
    logger.info(f"   Modelos fallidos: {len(failed_models)}/{len(results)}")
    
    return successful_models, failed_models

def save_training_report(results, output_path='results/training_report.json'):
    """Guardar reporte detallado del entrenamiento"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(results),
        'successful_models': len([r for r in results if r['status'] == 'success']),
        'failed_models': len([r for r in results if r['status'] == 'error']),
        'results': results
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Reporte guardado en: {output_path}")

def verify_saved_models(results, models_dir='models/'):
    """Verificar que los modelos se guardaron correctamente"""
    logger.info("\n🔍 Verificando modelos guardados...")
    
    for result in results:
        if result['status'] == 'success':
            model_path = result['model_path']
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / 1024  # KB
                logger.info(f"✓ {result['model_name']}: {file_size:.1f} KB")
            else:
                logger.error(f"❌ {result['model_name']}: Archivo no encontrado")

def main():
    """Función principal"""
    try:
        logger.info("🧠 ENTRENAMIENTO DISTRIBUIDO DE MODELOS ML")
        logger.info("Predicción de Déficit Energético")
        logger.info("="*50)
        
        # Configuración
        data_path = 'data/energia_dataset.csv'  # Ajusta la ruta según tu estructura
        
        # Preparar entorno
        setup_directories()
        
        # Validar dataset
        validate_dataset(data_path)
        
        # Inicializar Ray
        initialize_ray()
        
        # Ejecutar entrenamiento distribuido
        start_time = datetime.now()
        results = run_distributed_training(data_path)
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Analizar resultados
        successful_models, failed_models = analyze_results(results)
        
        # Verificar archivos guardados
        verify_saved_models(results)
        
        # Guardar reporte
        save_training_report(results)
        
        logger.info(f"\n⏱️  TIEMPO TOTAL: {total_time:.2f} segundos")
        logger.info("🎉 Entrenamiento distribuido completado!")
        
        return len(successful_models) > 0
        
    except Exception as e:
        logger.error(f"❌ Error en entrenamiento: {str(e)}")
        return False
    
    finally:
        # Limpiar Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray finalizado")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
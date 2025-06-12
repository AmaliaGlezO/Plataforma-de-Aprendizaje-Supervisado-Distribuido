#!/usr/bin/env python3
"""
Script para probar la API de predicción de déficit energético
Incluye pruebas automáticas y ejemplos de uso
"""

import requests
import json
import time
import logging
from typing import Dict, List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la API
API_BASE_URL = "http://localhost:8000"

# Datos de ejemplo para pruebas
SAMPLE_DATA = {
    "disponibilidad": 2235.0,
    "demanda_maxima": 3170.0,
    "afectacion": 1005.0,
    "respaldo": 0,
    "horario_pico": 10,
    "unidades_averia": 4,
    "unidades_mantenimiento": 0,
    "limitacion_termica": 313.0,
    "motores_impacto": 918.0,
    "year": 2022,
    "month": 12
}

BATCH_DATA = {
    "data": [
        SAMPLE_DATA,
        {
            "disponibilidad": 2419.0,
            "demanda_maxima": 3080.0,
            "afectacion": 731.0,
            "respaldo": 0,
            "horario_pico": 10,
            "unidades_averia": 5,
            "unidades_mantenimiento": 2,
            "limitacion_termica": 271.0,
            "motores_impacto": 915.0,
            "year": 2022,
            "month": 12
        },
        {
            "disponibilidad": 2511.0,
            "demanda_maxima": 2900.0,
            "afectacion": 659.0,
            "respaldo": 1,
            "horario_pico": 8,
            "unidades_averia": 3,
            "unidades_mantenimiento": 2,
            "limitacion_termica": 273.0,
            "motores_impacto": 902.0,
            "year": 2022,
            "month": 12
        }
    ]
}

def test_api_health():
    """Probar health check de la API"""
    logger.info("🔍 Probando health check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ API saludable: {data['status']}")
            logger.info(f"   Modelos cargados: {data['models_loaded']}")
            logger.info(f"   Total modelos: {data['total_models']}")
            return True
        else:
            logger.error(f"❌ Health check falló: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error conectando a la API: {str(e)}")
        return False

def test_list_models():
    """Probar listado de modelos"""
    logger.info("📋 Probando listado de modelos...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json()
            logger.info(f"✅ Modelos encontrados: {len(models)}")
            for model in models:
                logger.info(f"   - {model}")
            return models
        else:
            logger.error(f"❌ Error listando modelos: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"❌ Error listando modelos: {str(e)}")
        return []

def test_model_info(model_name: str):
    """Probar información detallada de un modelo"""
    logger.info(f"ℹ️  Probando info del modelo: {model_name}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models/{model_name}")
        if response.status_code == 200:
            info = response.json()
            logger.info(f"✅ Información del modelo {model_name}:")
            logger.info(f"   Tipo: {info['type']}")
            logger.info(f"   Tamaño: {info['file_size_kb']:.2f} KB")
            logger.info(f"   Entrenado: {info['trained_at']}")
            logger.info(f"   En memoria: {info['is_loaded']}")
            if info['metrics']:
                logger.info(f"   Test R²: {info['metrics'].get('test_r2', 'N/A')}")
                logger.info(f"   Test MAE: {info['metrics'].get('test_mae', 'N/A')}")
            return info
        else:
            logger.error(f"❌ Error obteniendo info: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Error obteniendo info: {str(e)}")
        return None

def test_load_model(model_name: str):
    """Probar carga de modelo en memoria"""
    logger.info(f"💾 Probando carga del modelo: {model_name}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/models/{model_name}/load")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Modelo cargado: {data['message']}")
            return True
        else:
            logger.error(f"❌ Error cargando modelo: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {str(e)}")
        return False

def test_prediction(model_name: str, data: Dict):
    """Probar predicción individual"""
    logger.info(f"🔮 Probando predicción con modelo: {model_name}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/{model_name}",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Predicción exitosa:")
            logger.info(f"   Déficit predicho: {result['predicted_deficit']:.2f} MW")
            logger.info(f"   Tiempo de procesamiento: {result['processing_time_ms']:.2f} ms")
            return result
        else:
            logger.error(f"❌ Error en predicción: {response.status_code}")
            logger.error(f"   Respuesta: {response.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        return None

def test_batch_prediction(model_name: str, batch_data: Dict):
    """Probar predicción en lote"""
    logger.info(f"📊 Probando predicción en lote con modelo: {model_name}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/{model_name}/batch",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Predicción en lote exitosa:")
            logger.info(f"   Muestras procesadas: {result['total_samples']}")
            logger.info(f"   Tiempo total: {result['processing_time_ms']:.2f} ms")
            logger.info(f"   Predicciones: {[f'{p:.2f}' for p in result['predictions']]}")
            return result
        else:
            logger.error(f"❌ Error en predicción batch: {response.status_code}")
            logger.error(f"   Respuesta: {response.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Error en predicción batch: {str(e)}")
        return None

def test_cache_status():
    """Probar estado del cache"""
    logger.info("🗂️  Probando estado del cache...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/cache/status")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Estado del cache:")
            logger.info(f"   Modelos en cache: {data['cached_models']}")
            logger.info(f"   Tamaño del cache: {data['cache_size']}")
            return data
        else:
            logger.error(f"❌ Error obteniendo estado del cache: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado del cache: {str(e)}")
        return None

def run_performance_test(model_name: str, num_requests: int = 10):
    """Ejecutar prueba de rendimiento"""
    logger.info(f"⚡ Ejecutando prueba de rendimiento: {num_requests} requests...")
    
    times = []
    successful_requests = 0
    
    for i in range(num_requests):
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/{model_name}",
                json=SAMPLE_DATA,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                successful_requests += 1
                request_time = (time.time() - start_time) * 1000  # ms
                times.append(request_time)
        except Exception as e:
            logger.error(f"Error en request {i+1}: {str(e)}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        logger.info(f"📈 Resultados de rendimiento:")
        logger.info(f"   Requests exitosos: {successful_requests}/{num_requests}")
        logger.info(f"   Tiempo promedio: {avg_time:.2f} ms")
        logger.info(f"   Tiempo mínimo: {min_time:.2f} ms")
        logger.info(f"   Tiempo máximo: {max_time:.2f} ms")
        logger.info(f"   Requests/segundo: {1000/avg_time:.2f}")

def generate_curl_examples(models: List[str]):
    """Generar ejemplos de comandos curl"""
    logger.info("\n🌐 EJEMPLOS DE COMANDOS CURL:")
    logger.info("="*50)
    
    # Health check
    logger.info("# Health check")
    logger.info(f"curl -X GET {API_BASE_URL}/health")
    
    # Listar modelos
    logger.info("\n# Listar modelos")
    logger.info(f"curl -X GET {API_BASE_URL}/models")
    
    if models:
        model_name = models[0]
        
        # Info del modelo
        logger.info(f"\n# Información del modelo")
        logger.info(f"curl -X GET {API_BASE_URL}/models/{model_name}")
        
        # Cargar modelo
        logger.info(f"\n# Cargar modelo en memoria")
        logger.info(f"curl -X POST {API_BASE_URL}/models/{model_name}/load")
        
        # Predicción
        logger.info(f"\n# Predicción individual")
        curl_data = json.dumps(SAMPLE_DATA)
        logger.info(f"curl -X POST {API_BASE_URL}/predict/{model_name} \\")
        logger.info(f"  -H 'Content-Type: application/json' \\")
        logger.info(f"  -d '{curl_data}'")
        
        # Predicción en lote
        logger.info(f"\n# Predicción en lote")
        curl_batch_data = json.dumps(BATCH_DATA)
        logger.info(f"curl -X POST {API_BASE_URL}/predict/{model_name}/batch \\")
        logger.info(f"  -H 'Content-Type: application/json' \\")
        logger.info(f"  -d '{curl_batch_data}'")

def main():
    """Función principal de pruebas"""
    logger.info("🧪 PRUEBAS DE LA API DE PREDICCIÓN ENERGÉTICA")
    logger.info("="*55)
    
    # Verificar que la API esté disponible
    if not test_api_health():
        logger.error("API no disponible. Asegúrate de que esté ejecutándose.")
        return False
    
    # Listar modelos disponibles
    models = test_list_models()
    if not models:
        logger.warning("No hay modelos disponibles para probar.")
        logger.info("Ejecuta primero: python scripts/train_and_deploy.py")
        return False
    
    # Probar con el primer modelo disponible
    test_model = models[0]
    logger.info(f"\n🎯 Probando con modelo: {test_model}")
    
    # Probar información del modelo
    test_model_info(test_model)
    
    # Cargar modelo en memoria
    test
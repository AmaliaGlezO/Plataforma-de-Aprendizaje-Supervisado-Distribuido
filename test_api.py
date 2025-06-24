import requests
import json
from pprint import pprint

API_BASE_URL= 'http://ml-api:8000'


def test_api():
    try:
        # 1. Verificar salud del servicio
        print("\n=== Health Check ===")
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        health_response.raise_for_status()  # Lanza excepción para códigos 4xx/5xx
        health = health_response.json()
        pprint(health)
        
        # 2. Listar modelos disponibles (corregido)
        print("\n=== Available Models ===")
        # Primero verificamos si el endpoint /models existe
        try:
            models_response = requests.get(f"{API_BASE_URL}/models", timeout=5)
            if models_response.status_code == 200:
                models = models_response.json()
                if isinstance(models, dict) and 'models' in models:
                    print("Modelos disponibles:", list(models['models'].keys()))
                else:
                    print("Respuesta inesperada de /models:", models)
            else:
                print(f"Endpoint /models no disponible (HTTP {models_response.status_code})")
                # Alternativa: listar archivos en directorio models
                print("Buscando modelos en directorio local...")
                try:
                    import os
                    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
                    print("Modelos encontrados (.pkl):", [f.replace('.pkl', '') for f in model_files])
                except Exception as e:
                    print("No se pudo leer directorio models:", str(e))
        except requests.exceptions.RequestException as e:
            print("Error al conectar con /models:", str(e))

        # 3. Realizar predicción (con validación)
        print("\n=== Making Prediction ===")
        prediction_data = {
            "model_name": "Ridge",  
            "features": [[2025, 6, 1000, 500, 0, 0, 0, 0, 0, 0, 0,0, 0, 0]]
        }
        
        try:
            prediction_response = requests.post(
                f"{API_BASE_URL}/predict",
                json=prediction_data,
                timeout=10
            )
            prediction_response.raise_for_status()
            prediction = prediction_response.json()
            pprint(prediction)
        except requests.exceptions.HTTPError as e:
            print(f"Error en predicción (HTTP {e.response.status_code}):")
            if e.response.status_code == 404:
                print("Modelo no encontrado. Prueba con uno de estos:")
                if 'models' in locals():
                    print(list(models['models'].keys()))
            print("Respuesta completa:", e.response.json())
        except json.JSONDecodeError:
            print("Respuesta no es JSON válido:", prediction_response.text)

    except requests.exceptions.ConnectionError:
        print(f"\nError: No se pudo conectar a la API en {API_BASE_URL}")
        print("Asegúrate que el servicio ml-api esté corriendo y el puerto 8000 esté accesible")
    except requests.exceptions.Timeout:
        print("\nError: Timeout al conectar con la API")
    except Exception as e:
        print("\nError inesperado:", str(e))

if __name__ == "__main__":
    test_api()
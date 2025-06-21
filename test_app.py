"""
Aplicación de prueba básica para verificar que todo funciona
"""
import ray
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="ML Platform Test", version="0.1.0")

@ray.remote
def test_ray_task(x):
    """Tarea simple de Ray para probar"""
    return x * x

@app.get("/")
async def root():
    return {"message": "🚀 ML Platform funcionando!", "status": "OK"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/test-ray")
async def test_ray():
    """Probar Ray"""
    try:
        # Inicializar Ray si no está inicializado
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Ejecutar tarea simple
        future = test_ray_task.remote(5)
        result = ray.get(future)
        
        return {
            "ray_status": "working",
            "test_result": result,
            "expected": 25,
            "success": result == 25
        }
    except Exception as e:
        return {
            "ray_status": "error",
            "error": str(e)
        }

@app.get("/info")
async def info():
    """Información del sistema"""
    info_data = {
        "ray_initialized": ray.is_initialized(),
        "python_path": os.environ.get("PYTHONPATH"),
        "working_dir": os.getcwd()
    }
    
    if ray.is_initialized():
        info_data["ray_cluster_resources"] = ray.cluster_resources()
    
    return info_data

if __name__ == "__main__":
    print("🔧 Iniciando aplicación de prueba...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
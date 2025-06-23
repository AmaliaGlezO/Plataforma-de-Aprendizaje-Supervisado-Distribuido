import requests

# 1. Verificar salud
health = requests.get("http://localhost:8000/health").json()
print("Health:", health)

# 2. Listar modelos
models = requests.get("http://localhost:8000/models").json()
print("Models:", models["models"].keys())

# 3. Realizar predicción
prediction = requests.post(
    "http://localhost:8000/predict",
    json={
        "model_name": "ridge",
        "features": [[2025, 6, 1000, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    }
).json()
print("Prediction:", prediction)
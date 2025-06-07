from fastapi import FastAPI, Request
import joblib

app = FastAPI()

@app.get("/health")
def health_check():
    pass

@app.post("/predict")
def predict(request: Request):
    pass

def load_model(model_name):
    pass

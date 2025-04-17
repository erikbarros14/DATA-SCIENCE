# api.py
from fastapi import FastAPI
import uvicorn
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Carregar modelos
print("Carregando modelos...")
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Modelos carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    exit()

class Features(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(features: Features):
    try:
        # Converter para numpy array
        X = np.array(features.features).reshape(1, -1)
        
        # Pré-processamento
        X_scaled = scaler.transform(X)
        
        # Previsão
        pred = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[0]
        
        return {
            "prediction": le.inverse_transform(pred)[0],
            "probability": float(proba[pred[0]]),
            "model": type(model).__name__
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def home():
    return {"message": "API de classificação de câncer está rodando"}

if __name__ == "__main__":
    print("\nIniciando servidor...")
    print("Acesse: http://127.0.0.1:8000")
    print("Documentação: http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
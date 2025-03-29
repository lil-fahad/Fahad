
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import torch

app = FastAPI()

# Placeholder model/scaler load — adjust paths
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                pred = model(input_tensor).cpu().numpy().flatten()
        else:
            pred = model.predict(X_scaled)
        return {"prediction": pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

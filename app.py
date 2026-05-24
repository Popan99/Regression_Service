import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import uvicorn
import os
import sklearn

app = FastAPI(title="ML Prediction API")

# Путь к модели (относительный, внутри контейнера)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_pipeline.joblib")

try:
    full_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    full_pipeline = None

@app.get("/health")
def health_check() -> Dict[str, str]:
    if full_pipeline is not None:
        return {"status": "ok", "message": "Model is loaded and ready"}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict")
async def predict(features: Dict[str, Any]):
    if full_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_df = pd.DataFrame([features])
        if hasattr(full_pipeline, 'feature_names_in_'):
            expected_cols = set(full_pipeline.feature_names_in_)
            missing = expected_cols - set(input_df.columns)
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        prediction = full_pipeline.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


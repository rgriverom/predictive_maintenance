import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict

import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / 'model_export'

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not EXPORT_DIR.exists():
        raise RuntimeError(f'{EXPORT_DIR} not found — run export_model.py first')

    model_state['model'] = mlflow.sklearn.load_model(str(EXPORT_DIR / 'model'))
    with open(EXPORT_DIR / 'feature_cols.json') as f:
        model_state['feature_cols'] = json.load(f)

    print(f'Loaded exported model, expecting {len(model_state["feature_cols"])} features')
    yield
    model_state.clear()


app = FastAPI(title="CMAPSS RUL Prediction API", lifespan=lifespan)


class SensorReading(BaseModel):
    readings: Dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.post("/predict")
def predict(reading: SensorReading):
    feature_cols = model_state['feature_cols']
    missing = [c for c in feature_cols if c not in reading.readings]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing sensor values: {missing}')

    row = pd.DataFrame([[reading.readings[c] for c in feature_cols]], columns=feature_cols)
    predicted_rul = model_state['model'].predict(row)[0]
    return {"predicted_rul_cycles": float(predicted_rul)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
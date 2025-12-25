import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.path.join("models", "nba_rf_model.joblib")

app = FastAPI(title="NBA Outcome Predictor", version="1.0.0")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first: python src/train.py")

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_cols = _bundle["feature_columns"]


class PredictRequest(BaseModel):
    home_elo: float
    away_elo: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    home_elo = float(req.home_elo)
    away_elo = float(req.away_elo)

    df = pd.DataFrame([{
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
    }])[_cols]

    p_home = float(_model.predict_proba(df)[0][1])
    pred = int(p_home >= 0.5)
    return {"p_home_win": p_home, "prediction_home_win": pred}

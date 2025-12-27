import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.path.join("models", "nba_rf_model.joblib")

# Create app FIRST
app = FastAPI(title="NBA Outcome Predictor", version="1.0.0")

# Load model bundle at import-time
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first: python src/train.py")

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_cols = _bundle["feature_columns"]


class PredictRequest(BaseModel):
    home_elo: float
    away_elo: float

    # Optional extras (safe defaults)
    home_rest_days: float = 0.0
    away_rest_days: float = 0.0
    home_injury_impact: float = 0.0
    away_injury_impact: float = 0.0
    home_recent_winrate: float = 0.5
    away_recent_winrate: float = 0.5
    home_win_prob: float = 0.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    row = {
        "home_elo": float(req.home_elo),
        "away_elo": float(req.away_elo),
        "elo_diff": float(req.home_elo) - float(req.away_elo),

        "home_rest_days": float(req.home_rest_days),
        "away_rest_days": float(req.away_rest_days),
        "home_injury_impact": float(req.home_injury_impact),
        "away_injury_impact": float(req.away_injury_impact),
        "home_recent_winrate": float(req.home_recent_winrate),
        "away_recent_winrate": float(req.away_recent_winrate),
        "home_win_prob": float(req.home_win_prob),
    }

    df = pd.DataFrame([row])

    # Ensure any columns the model expects exist
    for c in _cols:
        if c not in df.columns:
            df[c] = 0.0

    df = df[_cols]
    p_home = float(_model.predict_proba(df)[0][1])
    return {"p_home_win": p_home, "prediction_home_win": int(p_home >= 0.5)}

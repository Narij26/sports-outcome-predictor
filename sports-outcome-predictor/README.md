# ML-Based NBA Outcome Predictor (Real Data)

End-to-end machine learning project to predict NBA game outcomes using **real historical NBA data** from FiveThirtyEight's public NBA Elo dataset.

## Features
- Real data ingestion (FiveThirtyEight NBA Elo CSV)
- Reproducible preprocessing + feature engineering
- Model training + evaluation (Random Forest)
- Saved model artifact (`joblib`)
- Deployable FastAPI service (`/predict`)

## Data Source
This project downloads real historical NBA game data from FiveThirtyEight's NBA Elo model dataset:
- https://projects.fivethirtyeight.com/nba-model/

## Quickstart

### 1) Create venv + install deps
```bash
python -m venv .venv
# mac/linux:
source .venv/bin/activate
# windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Download real NBA data
```bash
python src/fetch_data.py
```

### 3) Train and evaluate
```bash
python src/train.py
```

### 4) Run a local prediction
```bash
python src/predict.py --home_elo 1650 --away_elo 1580
```

### 5) Run the API
```bash
uvicorn src.api:app --reload --port 8000
```

Test:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_elo":1650,"away_elo":1580}'
```

## Notes
- The model uses Elo ratings as a strong baseline signal for team strength.
- You can extend features using `elo_prob1`, recent performance windows, rest days, etc.

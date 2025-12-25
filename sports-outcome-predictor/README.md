# ML-Based NBA Outcome Predictor (Synthetic Data)

End-to-end machine learning project that predicts NBA-style game outcomes using
**synthetically generated (mock) data** designed to closely resemble real-world
NBA conditions.

The project demonstrates a full ML pipeline — data generation, preprocessing,
model training, evaluation, and API-based inference — with a strong emphasis on
reproducibility and clean system design.

---

## Features
- Realistic **synthetic NBA-style data generation**
- Encodes team strength, home-court advantage, rest days, injuries, and recent form
- Reproducible preprocessing + feature engineering
- Model training + evaluation (Random Forest)
- Saved model artifact (`joblib`)
- Deployable FastAPI service (`/predict`)

---

## Data Source

This project intentionally uses **synthetically generated (mock) NBA-style game
data** to ensure full reproducibility and independence from third-party APIs.

Data is generated locally using:

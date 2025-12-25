import argparse
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "nba_rf_model.joblib")


def predict_home_win_proba(home_elo: float, away_elo: float) -> float:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    cols = bundle["feature_columns"]

    df = pd.DataFrame([{
        "home_elo": float(home_elo),
        "away_elo": float(away_elo),
        "elo_diff": float(home_elo) - float(away_elo),
    }])[cols]

    return float(model.predict_proba(df)[0][1])


def main():
    p = argparse.ArgumentParser(description="Predict P(home win) from pre-game Elo ratings.")
    p.add_argument("--home_elo", type=float, required=True)
    p.add_argument("--away_elo", type=float, required=True)
    args = p.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train first: python src/train.py")

    proba = predict_home_win_proba(args.home_elo, args.away_elo)
    print(f"P(home win) = {proba:.3f}")


if __name__ == "__main__":
    main()

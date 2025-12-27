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
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,

        # defaults for optional features
        "home_rest_days": 0.0,
        "away_rest_days": 0.0,
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "home_recent_winrate": 0.5,
        "away_recent_winrate": 0.5,
        "home_win_prob": 0.5,
    }])

    # ensure all required columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

    df = df[cols]
    return float(model.predict_proba(df)[0][1])


def main():
    parser = argparse.ArgumentParser(
        description="Predict probability of home team winning"
    )
    parser.add_argument("--home_elo", type=float, required=True)
    parser.add_argument("--away_elo", type=float, required=True)
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run: python src/train.py"
        )

    home_elo = args.home_elo
    away_elo = args.away_elo

    p = predict_home_win_proba(home_elo, away_elo)
    print(f"P(home win) = {p:.3f}")


if __name__ == "__main__":
    main()

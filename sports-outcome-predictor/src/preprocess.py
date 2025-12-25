import pandas as pd

REQUIRED_COLS = [
    "elo1_pre", "elo2_pre",
    "score1", "score2",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 538 NBA Elo dataset into ML-ready features and target."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from dataset: {missing}")

    df = df.copy()
    df["home_win"] = (df["score1"] > df["score2"]).astype(int)

    # Leakage-safe baseline: use pre-game Elo ratings only.
    out = pd.DataFrame({
        "home_elo": df["elo1_pre"].astype(float),
        "away_elo": df["elo2_pre"].astype(float),
    })
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["home_win"] = df["home_win"].astype(int)

    return out.dropna()

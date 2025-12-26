# src/preprocess.py
import pandas as pd

MOCK_REQUIRED = [
    "home_rating", "away_rating", "home_win"
]

MOCK_OPTIONAL = [
    "home_rest_days", "away_rest_days",
    "home_injury_impact", "away_injury_impact",
    "home_recent_winrate", "away_recent_winrate",
    "home_win_prob",
]

FTE_REQUIRED = ["elo1_pre", "elo2_pre", "score1", "score2"]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataset into ML-ready features + target.

    Supports:
    - Mock dataset from mock_data.py (home_rating/away_rating/.../home_win)
    - FiveThirtyEight-style dataset (elo1_pre/elo2_pre/score1/score2)
    """

    df = df.copy()

    # --- Case A: Mock dataset ---
    if all(c in df.columns for c in MOCK_REQUIRED):
        out = pd.DataFrame()
        out["home_elo"] = df["home_rating"].astype(float)
        out["away_elo"] = df["away_rating"].astype(float)
        out["elo_diff"] = out["home_elo"] - out["away_elo"]

        # Add optional columns if present
        for c in MOCK_OPTIONAL:
            if c in df.columns:
                out[c] = df[c].astype(float)

        out["home_win"] = df["home_win"].astype(int)
        return out.dropna()

    # --- Case B: FiveThirtyEight dataset ---
    if all(c in df.columns for c in FTE_REQUIRED):
        out = pd.DataFrame()
        out["home_elo"] = df["elo1_pre"].astype(float)
        out["away_elo"] = df["elo2_pre"].astype(float)
        out["elo_diff"] = out["home_elo"] - out["away_elo"]
        out["home_win"] = (df["score1"] > df["score2"]).astype(int)
        return out.dropna()

    raise ValueError(
        "Unrecognized dataset schema. "
        "Expected mock columns (home_rating, away_rating, home_win, ...) "
        "or 538 columns (elo1_pre, elo2_pre, score1, score2)."
    )

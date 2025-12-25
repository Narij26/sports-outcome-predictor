import os
import pandas as pd

FIVETHIRTYEIGHT_URL = "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"
OUT_PATH = os.path.join("data", "nba_games.csv")


def fetch_nba_data(min_season: int = 2010) -> pd.DataFrame:
    """Download FiveThirtyEight NBA Elo dataset and save to data/nba_games.csv."""
    os.makedirs("data", exist_ok=True)
    print("Downloading FiveThirtyEight NBA Elo dataset...")
    df = pd.read_csv(FIVETHIRTYEIGHT_URL)

    # Keep modern seasons by default
    if "season" in df.columns:
        df = df[df["season"] >= min_season].copy()

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved dataset -> {OUT_PATH} (rows={len(df)})")
    return df


if __name__ == "__main__":
    fetch_nba_data()

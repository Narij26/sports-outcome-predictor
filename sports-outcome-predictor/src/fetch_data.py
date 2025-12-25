import pandas as pd

def load_data(path="data/mock_games.csv"):
    """
    Load synthetically generated NBA-style game data.
    """
    return pd.read_csv(path)

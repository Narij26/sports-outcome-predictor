# src/predict.py (replace df creation)

df = pd.DataFrame([{
    "home_elo": float(home_elo),
    "away_elo": float(away_elo),
    "elo_diff": float(home_elo) - float(away_elo),

    "home_rest_days": 0.0,
    "away_rest_days": 0.0,
    "home_injury_impact": 0.0,
    "away_injury_impact": 0.0,
    "home_recent_winrate": 0.5,
    "away_recent_winrate": 0.5,
    "home_win_prob": 0.5,
}])

for c in cols:
    if c not in df.columns:
        df[c] = 0.0

df = df[cols]

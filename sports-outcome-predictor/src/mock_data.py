import csv
import random
from datetime import datetime, timedelta
from collections import defaultdict, deque

TEAMS = ["LAL","BOS","GSW","PHX","MIL","DEN","MIA","DAL","NYK","CHI"]

def logistic(x: float) -> float:
    # Stable-ish logistic for converting a score to probability
    return 1.0 / (1.0 + (2.718281828 ** (-x)))

def generate_mock_games(
    seasons=2,
    games_per_season=600,
    seed=42,
    base_date=datetime(2022, 10, 1),
):
    random.seed(seed)

    # "True strength" for each team (latent), with small drift across time
    team_base = {t: random.randint(1480, 1620) for t in TEAMS}

    # Track last played date to compute rest days
    last_played = {t: None for t in TEAMS}

    # Track rolling form: last N game results (1=win,0=loss)
    form_window = 10
    recent_results = {t: deque(maxlen=form_window) for t in TEAMS}

    rows = []
    current_date = base_date

    for season in range(1, seasons + 1):
        # Offseason: small random reshuffle
        for t in TEAMS:
            team_base[t] += random.randint(-25, 25)

        # Reset last played at season start (keeps rest meaningful)
        last_played = {t: None for t in TEAMS}
        for t in TEAMS:
            recent_results[t].clear()

        for i in range(games_per_season):
            home, away = random.sample(TEAMS, 2)

            # Schedule gaps: most days have games; sometimes skip a day
            if random.random() < 0.15:
                current_date += timedelta(days=1)

            # Rest days (cap at 7 for realism)
            def rest_days(team):
                lp = last_played[team]
                if lp is None:
                    return 3  # season opener-ish default
                return min((current_date - lp).days, 7)

            home_rest = rest_days(home)
            away_rest = rest_days(away)

            # Injuries: random games where key players are out (bigger hit)
            # Keep it simple: injury impact is 0–50 points, more often small.
            def injury_impact():
                if random.random() < 0.12:  # ~12% chance some injury impact
                    return random.choice([10, 15, 20, 25, 35, 50])
                return 0

            home_injury = injury_impact()
            away_injury = injury_impact()

            # Rolling form (recent win rate). If no games yet, neutral 0.5
            def winrate(team):
                if len(recent_results[team]) == 0:
                    return 0.5
                return sum(recent_results[team]) / len(recent_results[team])

            home_form = winrate(home)  # 0..1
            away_form = winrate(away)

            # Build “effective rating” from:
            # base strength + home-court + rest advantage - injury + form boost
            home_court = 55  # Elo-ish home advantage points

            # Rest effect: each rest day above opponent adds a small boost
            rest_effect = 6 * (home_rest - away_rest)  # points

            # Form effect: map form (0..1) to roughly (-25..+25)
            form_effect = 50 * (home_form - away_form)

            home_rating = team_base[home] + home_court + rest_effect + form_effect - home_injury
            away_rating = team_base[away] - rest_effect - form_effect - away_injury

            # Probability from Elo difference
            # (difference/400) gives standard Elo curve; add tiny randomness
            elo_diff = home_rating - away_rating
            prob_home_win = 1 / (1 + 10 ** (-elo_diff / 400))
            prob_home_win = min(max(prob_home_win + random.uniform(-0.03, 0.03), 0.01), 0.99)

            home_win = 1 if random.random() < prob_home_win else 0

            # Scores: loosely tied to ratings + noise
            # Keep scores NBA-ish (90–140)
            base_points = 108
            home_score = int(base_points + (elo_diff / 50) + random.gauss(0, 8))
            away_score = int(base_points - (elo_diff / 50) + random.gauss(0, 8))
            home_score = max(80, min(home_score, 150))
            away_score = max(80, min(away_score, 150))

            # Update rolling results
            recent_results[home].append(home_win)
            recent_results[away].append(1 - home_win)

            # Update last played
            last_played[home] = current_date
            last_played[away] = current_date

            # Advance date most of the time
            current_date += timedelta(days=1 if random.random() < 0.75 else 0)

            rows.append([
                season,
                current_date.strftime("%Y-%m-%d"),
                home,
                away,
                int(home_rating),
                int(away_rating),
                home_rest,
                away_rest,
                home_injury,
                away_injury,
                round(home_form, 3),
                round(away_form, 3),
                round(prob_home_win, 4),
                home_score,
                away_score,
                home_win
            ])

        # Offseason gap
        current_date += timedelta(days=120)

    return rows

def write_csv(path="data/mock_games.csv"):
    rows = generate_mock_games()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "season",
            "date",
            "home_team",
            "away_team",
            "home_rating",
            "away_rating",
            "home_rest_days",
            "away_rest_days",
            "home_injury_impact",
            "away_injury_impact",
            "home_recent_winrate",
            "away_recent_winrate",
            "home_win_prob",
            "home_score",
            "away_score",
            "home_win"
        ])
        w.writerows(rows)

if __name__ == "__main__":
    write_csv()
    print("Wrote data/mock_games.csv")

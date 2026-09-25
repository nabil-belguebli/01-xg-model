"""Construction du tableau de tirs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from . import statsbomb
    from .features import shot_to_row
except ImportError:  # python src/dataset.py
    import statsbomb
    from features import shot_to_row

# (competition_id, season_id, libellé) — liste à jour : statsbomb.competitions()
COMPETITIONS = [
    (43, 3, "Coupe du monde 2018"),
    (43, 106, "Coupe du monde 2022"),
    (72, 30, "Coupe du monde féminine 2019"),
    (55, 43, "Euro 2020"),
]

OUT = Path(__file__).resolve().parent.parent / "data" / "shots.csv"


def build(competitions=COMPETITIONS, limit_matches: int | None = None) -> pd.DataFrame:
    rows = []
    for competition_id, season_id, label in competitions:
        games = statsbomb.matches(competition_id, season_id)
        if limit_matches:
            games = games[:limit_matches]
        print(f"{label} : {len(games)} matchs")

        for i, game in enumerate(games, 1):
            match_id = game["match_id"]
            for shot, key_pass in statsbomb.shots(match_id):
                row = shot_to_row(shot, match_id, key_pass)
                row["competition"] = label
                rows.append(row)
            if i % 10 == 0:
                print(f"  {i}/{len(games)} matchs, {len(rows)} tirs")

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} tirs, {df['is_goal'].sum()} buts ({100 * df['is_goal'].mean():.1f} %)")
    print(f"{df['has_freeze_frame'].mean() * 100:.0f} % avec freeze frame")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="matchs par compétition")
    args = parser.parse_args()

    df = build(limit_matches=args.limit)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"écrit dans {OUT}")

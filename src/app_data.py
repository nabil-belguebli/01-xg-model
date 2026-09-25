"""Prépare data/app_shots.csv pour l'application : un tir par ligne, avec xG et contributions.

Calculé une fois ici plutôt qu'à chaque ouverture de l'application.
"""

from __future__ import annotations

import pandas as pd

try:
    from .explain import out_of_fold_explanations
    from .train import DATA
except ImportError:  # python src/app_data.py
    from explain import out_of_fold_explanations
    from train import DATA

OUT = DATA.parent / "app_shots.csv"


def main():
    df = pd.read_csv(DATA)
    shots = df[df["shot_type"] != "Penalty"].reset_index(drop=True)
    explained = pd.concat([shots, out_of_fold_explanations(shots)], axis=1)
    explained.to_csv(OUT, index=False)
    print(f"{len(explained)} tirs, {explained['xg'].sum():.1f} xG pour "
          f"{explained['is_goal'].sum()} buts, écrit dans {OUT}")


if __name__ == "__main__":
    main()

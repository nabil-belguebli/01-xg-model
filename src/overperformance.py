"""Talent ou chance : probabilité qu'un écart buts − xG soit dû au hasard.

Hypothèse nulle : chaque tir est un tirage indépendant dont la probabilité de
but est son xG. Le nombre de buts suit alors une loi de Poisson-binomiale,
calculée exactement par programmation dynamique (pas de simulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def goals_distribution(xg) -> np.ndarray:
    """P(nombre de buts = k) pour k = 0..n, tirs indépendants de probabilités xg."""
    distribution = np.array([1.0])
    for p in np.asarray(xg, dtype=float):
        distribution = np.append(distribution * (1 - p), 0.0) + np.append(0.0, distribution * p)
    return distribution


def chance_probability(xg, goals: int) -> float:
    """Probabilité d'un écart au moins aussi grand, dans le même sens, par pur hasard.

    Sur-performance (buts > xG) : P(buts ≥ observé). Sous-performance : P(buts ≤ observé).
    """
    distribution = goals_distribution(xg)
    if goals >= np.sum(xg):
        return float(distribution[goals:].sum())
    return float(distribution[: goals + 1].sum())


def player_table(shots: pd.DataFrame, min_shots: int = 10) -> pd.DataFrame:
    """Un joueur par ligne : tirs, buts, xG, écart, probabilité que ce soit du hasard."""
    rows = []
    for (player, team), group in shots.groupby(["player", "team"]):
        if len(group) < min_shots:
            continue
        goals = int(group["is_goal"].sum())
        rows.append({
            "joueur": player,
            "equipe": team,
            "tirs": len(group),
            "buts": goals,
            "xg": group["xg"].sum(),
            "ecart": goals - group["xg"].sum(),
            "proba_hasard": chance_probability(group["xg"], goals),
        })
    return pd.DataFrame(rows).sort_values("ecart", ascending=False, ignore_index=True)


def expected_false_alarms(shots: pd.DataFrame, players: pd.DataFrame, threshold: float = 0.05) -> float:
    """Nombre de joueurs attendus sous le seuil si tous n'avaient que de la chance.

    Pour chaque joueur, on parcourt tous les nombres de buts possibles sous
    l'hypothèse du hasard, et on additionne la probabilité de ceux qui
    donneraient une probabilité sous le seuil. Exact, sans simulation.
    """
    expected = 0.0
    for player, team in zip(players["joueur"], players["equipe"]):
        xg = shots.loc[(shots["player"] == player) & (shots["team"] == team), "xg"]
        distribution = goals_distribution(xg)
        expected += sum(p for k, p in enumerate(distribution) if chance_probability(xg, k) < threshold)
    return expected

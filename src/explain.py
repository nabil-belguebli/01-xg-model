"""xG hors-pli de chaque tir, et ce que chaque feature y a apporté.

Pour une régression logistique, les valeurs SHAP ont une forme exacte :
contribution de la feature j = coef_j × (valeur_j − moyenne_j), sur l'échelle
des log-cotes. Point de départ (tir « moyen ») + somme des contributions =
log-cote du tir, que la sigmoïde transforme en xG. Pas besoin de la
bibliothèque shap, qui calculerait exactement la même chose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from .train import FEATURE_STEPS, build_model, choose_regularization
except ImportError:  # python src/explain.py
    from train import FEATURE_STEPS, build_model, choose_regularization

NUMERIC = [f for _, numeric, _ in FEATURE_STEPS for f in numeric]
CATEGORICAL = [f for _, _, categorical in FEATURE_STEPS for f in categorical]

LABELS = {
    "distance": "Distance au but",
    "angle": "Angle de tir",
    "body_part": "Partie du corps",
    "shot_type": "Type de tir",
    "technique": "Technique",
    "first_time": "Première intention",
    "under_pressure": "Sous pression",
    "defenders_in_triangle": "Défenseurs dans le triangle",
    "keeper_to_goal": "Gardien – but",
    "keeper_to_shooter": "Gardien – tireur",
    "nearest_defender": "Défenseur le plus proche",
    "assist_type": "Type de passe décisive",
    "assist_height": "Hauteur de la passe",
    "one_on_one": "Face-à-face",
    "open_goal": "But vide",
    "aerial_won": "Duel aérien gagné",
    "play_pattern": "Phase de jeu",
}


def sigmoid(logit):
    return 1.0 / (1.0 + np.exp(-np.asarray(logit)))


def _column_groups(model) -> list[str]:
    """Feature d'origine de chaque colonne après one-hot (body_part_Head -> body_part)."""
    groups = []
    for name in model.named_steps["prep"].get_feature_names_out():
        kind, column = name.split("__", 1)
        if kind == "num":
            groups.append(column)
        else:
            groups.append(next(f for f in CATEGORICAL if column.startswith(f + "_")))
    return groups


def contributions(model, train: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """Contribution de chaque feature (log-cotes) pour chaque tir, par rapport au tir moyen du train."""
    prep, logreg = model.named_steps["prep"], model.named_steps["logreg"]
    coef = logreg.coef_[0]
    mean = prep.transform(train).mean(axis=0)

    per_column = coef * (prep.transform(shots) - mean)
    per_feature = pd.DataFrame(per_column, index=shots.index).T.groupby(_column_groups(model)).sum().T

    out = per_feature[NUMERIC + CATEGORICAL].add_prefix("contrib__")
    out.insert(0, "base_logit", logreg.intercept_[0] + coef @ mean)
    return out


def out_of_fold_explanations(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Pour chaque tir : xG et contributions, calculés par le modèle du pli qui n'a pas vu son match.

    Mêmes plis et même régularisation que train.out_of_fold_xg : les xG sont identiques.
    """
    C = choose_regularization(NUMERIC, CATEGORICAL, df)
    parts = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(df, df["is_goal"], df["match_id"]):
        train, held_out = df.iloc[train_idx], df.iloc[test_idx]
        model = build_model(NUMERIC, CATEGORICAL).set_params(logreg__C=C).fit(train, train["is_goal"])
        part = contributions(model, train, held_out)
        part.insert(0, "xg", model.predict_proba(held_out)[:, 1])
        parts.append(part)
    return pd.concat(parts).loc[df.index]

"""Gradient boosting contre logistique, mêmes features, même test.

Hyperparamètres choisis par validation croisée groupée par match, sur le
train uniquement : le test n'est regardé qu'une fois, à la fin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold

try:
    from .train import (DATA, FEATURE_STEPS, REPORTS, build_model, calibration_table,
                        choose_regularization, evaluate, split_by_match)
except ImportError:  # python src/boosting.py
    from train import (DATA, FEATURE_STEPS, REPORTS, build_model, calibration_table,
                       choose_regularization, evaluate, split_by_match)

NUMERIC = [f for _, numeric, _ in FEATURE_STEPS for f in numeric]
CATEGORICAL = [f for _, _, categorical in FEATURE_STEPS for f in categorical]

# Peu de données (4 200 tirs) : la grille descend jusqu'aux souches (arbres à 2 feuilles).
PARAM_GRID = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_iter": [50, 100, 200, 400],
    "max_leaf_nodes": [2, 3, 7, 15],
    "min_samples_leaf": [20, 50, 100],
}


def to_boosting_input(df: pd.DataFrame, categories: dict[str, list]) -> pd.DataFrame:
    """Catégorielles en dtype category, avec les mêmes modalités en train et en test."""
    X = df[NUMERIC + CATEGORICAL].copy()
    for column in X.columns[X.dtypes == bool]:
        X[column] = X[column].astype(int)
    for column in CATEGORICAL:
        X[column] = pd.Categorical(X[column], categories=categories[column])
    return X


def tune(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> HistGradientBoostingClassifier:
    """Grille × 5 plis. Chaque pli laisse de côté des matchs entiers, comme le test."""
    search = GridSearchCV(
        HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0),
        PARAM_GRID,
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    print(f"\nMeilleurs hyperparamètres : {search.best_params_}")
    print(f"log loss en validation croisée : {-search.best_score_:.4f}")
    return search.best_estimator_


def bootstrap_by_match(test: pd.DataFrame, prob_a, prob_b, n: int = 2000, seed: int = 0) -> dict:
    """Écart de log loss (a - b) sur des jeux de test rééchantillonnés par match.

    On tire 57 matchs avec remise, n fois. Si l'intervalle à 95 % contient 0,
    l'écart observé peut n'être que du hasard.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"match_id": test["match_id"].to_numpy(), "y": test["is_goal"].to_numpy(),
                          "a": prob_a, "b": prob_b})
    by_match = [g for _, g in frame.groupby("match_id")]

    diffs = []
    for _ in range(n):
        sample = pd.concat([by_match[i] for i in rng.integers(0, len(by_match), len(by_match))])
        diffs.append(log_loss(sample["y"], sample["a"], labels=[0, 1])
                     - log_loss(sample["y"], sample["b"], labels=[0, 1]))
    diffs = np.array(diffs)
    return {
        "ecart_observe": log_loss(frame["y"], frame["a"]) - log_loss(frame["y"], frame["b"]),
        "ic95_bas": np.percentile(diffs, 2.5),
        "ic95_haut": np.percentile(diffs, 97.5),
        "part_a_meilleur": (diffs < 0).mean(),
    }


def main():
    df = pd.read_csv(DATA)
    open_play = df[df["shot_type"] != "Penalty"].copy()
    train, test = split_by_match(open_play)

    categories = {c: sorted(open_play[c].dropna().unique()) for c in CATEGORICAL}
    X_train, X_test = to_boosting_input(train, categories), to_boosting_input(test, categories)

    boosting = tune(X_train, train["is_goal"], train["match_id"])
    C = choose_regularization(NUMERIC, CATEGORICAL, train)
    logistic = build_model(NUMERIC, CATEGORICAL).set_params(logreg__C=C).fit(train, train["is_goal"])

    predictions = pd.DataFrame({
        "match_id": test["match_id"],
        "shot_id": test["shot_id"],
        "is_goal": test["is_goal"],
        "xg_baseline": build_model(FEATURE_STEPS[0][1], []).fit(train, train["is_goal"]).predict_proba(test)[:, 1],
        "xg_logistique": logistic.predict_proba(test)[:, 1],
        "xg_boosting": boosting.predict_proba(X_test)[:, 1],
        "xg_statsbomb": test["statsbomb_xg"],
    })

    results = {name: evaluate(predictions["is_goal"], predictions[column], name)
               for name, column in [("Baseline", "xg_baseline"),
                                    ("Logistique complète", "xg_logistique"),
                                    ("Gradient boosting", "xg_boosting"),
                                    ("xG StatsBomb", "xg_statsbomb")]}
    summary = pd.DataFrame(results).T
    print("\nRécapitulatif")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nÉcarts de log loss, bootstrap par match (négatif = le premier est meilleur)")
    comparisons = {}
    for a, b in [("xg_logistique", "xg_baseline"), ("xg_boosting", "xg_logistique"),
                 ("xg_boosting", "xg_statsbomb")]:
        result = bootstrap_by_match(test, predictions[a], predictions[b])
        comparisons[f"{a} - {b}"] = result
        print(f"  {a} - {b} : {result['ecart_observe']:+.4f}  "
              f"IC 95 % [{result['ic95_bas']:+.4f} ; {result['ic95_haut']:+.4f}]  "
              f"{a} meilleur dans {result['part_a_meilleur'] * 100:.0f} % des tirages")

    print("\nCalibration du gradient boosting")
    table = calibration_table(predictions["is_goal"], predictions["xg_boosting"])
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    REPORTS.mkdir(exist_ok=True)
    summary.to_csv(REPORTS / "metrics_modeles.csv", index_label="modele")
    pd.DataFrame(comparisons).T.to_csv(REPORTS / "bootstrap_log_loss.csv", index_label="comparaison")
    table.to_csv(REPORTS / "calibration_boosting.csv", index=False)
    predictions.to_csv(REPORTS / "predictions_test.csv", index=False)


if __name__ == "__main__":
    main()

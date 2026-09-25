"""Régression logistique : baseline distance + angle, puis ajout des features une à une."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "shots.csv"
REPORTS = Path(__file__).resolve().parent.parent / "reports"

BASELINE_FEATURES = ["distance", "angle"]

# Ajouts cumulatifs : chaque ligne garde les features des précédentes.
# (libellé, numériques, catégorielles)
FEATURE_STEPS = [
    ("distance + angle", BASELINE_FEATURES, []),
    ("+ partie du corps", [], ["body_part"]),
    ("+ type de tir", [], ["shot_type", "technique"]),
    ("+ première intention, pression", ["first_time", "under_pressure"], []),
    ("+ défenseurs dans le triangle", ["defenders_in_triangle"], []),
    ("+ position du gardien", ["keeper_to_goal", "keeper_to_shooter"], []),
    ("+ défenseur le plus proche", ["nearest_defender"], []),
    ("+ passe décisive", [], ["assist_type", "assist_height"]),
    ("+ face-à-face, but vide", ["one_on_one", "open_goal"], []),
    ("+ phase de jeu, duel aérien", ["aerial_won"], ["play_pattern"]),
]


def split_by_match(df: pd.DataFrame, test_fraction: float = 0.25, seed: int = 0):
    """Découpage par match, pas par tir : deux tirs d'un même match fuiteraient."""
    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)

    test_ids = set(match_ids[: int(len(match_ids) * test_fraction)])
    is_test = df["match_id"].isin(test_ids)
    return df[~is_test].copy(), df[is_test].copy()


def evaluate(y_true, y_prob, label: str) -> dict:
    """Pas d'accuracy : 10 % de buts, « jamais but » afficherait 90 %."""
    scores = {
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    print(f"\n{label}")
    print(f"  log loss : {scores['log_loss']:.4f}")
    print(f"  Brier    : {scores['brier']:.4f}")
    print(f"  ROC AUC  : {scores['roc_auc']:.4f}")
    return scores


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """xG prédit contre taux de buts réel, par tranche."""
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "xg_moyen_predit": predicted,
        "taux_de_buts_observe": observed,
        "ecart": observed - predicted,
    })


def build_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Numériques centrées-réduites, catégorielles en one-hot.

    Gardien hors du freeze frame (2 tirs) : distances remplacées par la médiane.
    Les catégories de moins de 20 tirs (lob, retourné...) sont regroupées :
    un coefficient estimé sur 3 tirs ne serait que du bruit.
    """
    columns = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                 ("scale", StandardScaler())]), numeric)]
    if categorical:
        columns.append(("cat", OneHotEncoder(min_frequency=20, handle_unknown="infrequent_if_exist",
                                             sparse_output=False), categorical))
    return Pipeline([
        ("prep", ColumnTransformer(columns)),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])


def cv_log_loss(model: Pipeline, train: pd.DataFrame) -> float:
    """Log loss moyen sur 5 plis du train, groupés par match. Le test n'est pas touché."""
    scores = cross_val_score(model, train, train["is_goal"], groups=train["match_id"],
                             cv=GroupKFold(n_splits=5), scoring="neg_log_loss")
    return -scores.mean()


def choose_regularization(numeric: list[str], categorical: list[str], train: pd.DataFrame) -> float:
    """C petit = coefficients retenus près de 0. Choisi par validation croisée, jamais sur le test."""
    search = GridSearchCV(build_model(numeric, categorical), {"logreg__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
                          scoring="neg_log_loss", cv=GroupKFold(n_splits=5))
    search.fit(train, train["is_goal"], groups=train["match_id"])
    return search.best_params_["logreg__C"]


def main():
    df = pd.read_csv(DATA)

    open_play = df[df["shot_type"] != "Penalty"].copy()  # penalty non pris en compte, tir différent
    print(f"{len(open_play)} tirs, {open_play['is_goal'].mean() * 100:.1f} % de buts")

    train, test = split_by_match(open_play)
    print(f"train : {len(train)} tirs / {train['match_id'].nunique()} matchs")
    print(f"test  : {len(test)} tirs / {test['match_id'].nunique()} matchs")

    has_reference = test["statsbomb_xg"].notna()  # référence à battre
    results = {}
    if has_reference.any():
        results["xG officiel StatsBomb"] = evaluate(test.loc[has_reference, "is_goal"],
                                                    test.loc[has_reference, "statsbomb_xg"],
                                                    "xG officiel StatsBomb")

    all_numeric = [f for _, numeric, _ in FEATURE_STEPS for f in numeric]
    all_categorical = [f for _, _, categorical in FEATURE_STEPS for f in categorical]
    C = choose_regularization(all_numeric, all_categorical, train)
    print(f"\nRégularisation retenue par validation croisée : C = {C}")

    numeric, categorical = [], []
    probabilities = {}
    for label, new_numeric, new_categorical in FEATURE_STEPS:
        numeric, categorical = numeric + new_numeric, categorical + new_categorical
        model = build_model(numeric, categorical).set_params(logreg__C=C)
        cv_score = cv_log_loss(model, train)
        model.fit(train, train["is_goal"])
        probabilities[label] = model.predict_proba(test)[:, 1]
        results[label] = {"log_loss_cv": cv_score,
                          **evaluate(test["is_goal"], probabilities[label], f"Logistique : {label}")}

    print("\nRécapitulatif (plus bas = mieux, sauf AUC). log_loss_cv : validation croisée sur le train")
    summary = pd.DataFrame(results).T
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    REPORTS.mkdir(exist_ok=True)
    summary.to_csv(REPORTS / "metrics_logistique.csv", index_label="modele")

    for filename, label in [("calibration_baseline.csv", FEATURE_STEPS[0][0]),
                            ("calibration_logistique.csv", FEATURE_STEPS[-1][0])]:
        print(f"\nCalibration : {label}")
        table = calibration_table(test["is_goal"], probabilities[label])
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        table.to_csv(REPORTS / filename, index=False)

    print("\nCoefficients du modèle complet (variables centrées-réduites)")
    names = model.named_steps["prep"].get_feature_names_out()
    for name, coefficient in zip(names, model.named_steps["logreg"].coef_[0]):
        print(f"  {name.split('__')[1]:<32} {coefficient:+.3f}")

if __name__ == "__main__":
    main()

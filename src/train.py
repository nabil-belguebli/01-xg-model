"""Baseline : régression logistique sur distance et angle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "shots.csv"
REPORTS = Path(__file__).resolve().parent.parent / "reports"

BASELINE_FEATURES = ["distance", "angle"]


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


def main():
    df = pd.read_csv(DATA)

    open_play = df[df["shot_type"] != "Penalty"].copy()  # penalty non pris en compte, tir différent
    print(f"{len(open_play)} tirs, {open_play['is_goal'].mean() * 100:.1f} % de buts")

    train, test = split_by_match(open_play)
    print(f"train : {len(train)} tirs / {train['match_id'].nunique()} matchs")
    print(f"test  : {len(test)} tirs / {test['match_id'].nunique()} matchs")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    model.fit(train[BASELINE_FEATURES], train["is_goal"])

    y_prob = model.predict_proba(test[BASELINE_FEATURES])[:, 1]
    evaluate(test["is_goal"], y_prob, "Baseline logistique (distance + angle)")

    has_reference = test["statsbomb_xg"].notna()  # référence à battre
    if has_reference.any():
        evaluate(test.loc[has_reference, "is_goal"],
                 test.loc[has_reference, "statsbomb_xg"],
                 "xG officiel StatsBomb")

    print("\nCalibration de la baseline")
    table = calibration_table(test["is_goal"], y_prob)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "calibration_baseline.csv", index=False)

    print("\nCoefficients (variables centrées-réduites)")
    for name, coefficient in zip(BASELINE_FEATURES, model.named_steps["logreg"].coef_[0]):
        print(f"  {name:<10} {coefficient:+.3f}")  # distance < 0, angle > 0 attendus


if __name__ == "__main__":
    main()

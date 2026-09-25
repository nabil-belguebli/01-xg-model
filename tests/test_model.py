"""Explications et test du hasard. Données synthétiques : data/ n'est pas versionné."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explain import CATEGORICAL, NUMERIC, contributions, out_of_fold_explanations, sigmoid  # noqa: E402
from overperformance import (chance_probability, expected_false_alarms,  # noqa: E402
                             goals_distribution)
from train import build_model, out_of_fold_xg  # noqa: E402


def synthetic_shots(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in NUMERIC})
    for f in ["first_time", "under_pressure", "one_on_one", "open_goal", "aerial_won"]:
        df[f] = rng.random(n) < 0.2
    for f in CATEGORICAL:
        df[f] = rng.choice(["a", "b", "c"], size=n)
    df["match_id"] = rng.integers(0, 40, size=n)
    df["is_goal"] = (rng.random(n) < sigmoid(-2 + df["angle"] - df["distance"])).astype(int)
    return df


def test_contributions_retombent_sur_lxg():
    """Tir moyen + somme des contributions = xG prédit par le modèle."""
    df = synthetic_shots()
    model = build_model(NUMERIC, CATEGORICAL).fit(df, df["is_goal"])
    out = contributions(model, df, df)
    logit = out["base_logit"] + out.filter(like="contrib__").sum(axis=1)
    assert np.allclose(sigmoid(logit), model.predict_proba(df)[:, 1])


def test_une_contribution_par_feature_dorigine():
    df = synthetic_shots()
    model = build_model(NUMERIC, CATEGORICAL).fit(df, df["is_goal"])
    columns = contributions(model, df, df).filter(like="contrib__").columns
    assert list(columns) == [f"contrib__{f}" for f in NUMERIC + CATEGORICAL]


def test_explications_hors_pli_coherentes_avec_train():
    df = synthetic_shots()
    assert np.allclose(out_of_fold_explanations(df)["xg"], out_of_fold_xg(df))


def test_loi_des_buts_deux_tirs_a_50():
    assert np.allclose(goals_distribution([0.5, 0.5]), [0.25, 0.5, 0.25])


def test_loi_des_buts_somme_a_un():
    assert math.isclose(goals_distribution(np.linspace(0.01, 0.6, 50)).sum(), 1.0)


def test_hasard_sur_performance():
    """3 tirs à 0,1 et 3 buts : 0,1³."""
    assert math.isclose(chance_probability([0.1, 0.1, 0.1], 3), 0.001)


def test_hasard_sous_performance():
    """10 tirs à 0,3 et aucun but : 0,7¹⁰."""
    assert math.isclose(chance_probability([0.3] * 10, 0), 0.7 ** 10)


def test_fausses_alertes_un_tir_a_50():
    """Un seul tir à 0,5 : but ou non, la probabilité vaut 0,5, jamais sous 5 %."""
    shots = pd.DataFrame({"player": ["A"], "team": ["X"], "xg": [0.5]})
    players = pd.DataFrame({"joueur": ["A"], "equipe": ["X"]})
    assert expected_false_alarms(shots, players) == 0.0


def test_fausses_alertes_sous_le_niveau_nominal():
    """Sous le hasard, au plus ~5 % par sens : jamais plus de 10 % des joueurs."""
    rng = np.random.default_rng(1)
    shots = pd.DataFrame({"player": np.repeat(list("ABCDEFGHIJ"), 40), "team": "X",
                          "xg": rng.uniform(0.02, 0.4, 400)})
    players = pd.DataFrame({"joueur": list("ABCDEFGHIJ"), "equipe": "X"})
    assert 0 < expected_false_alarms(shots, players) <= 1.0


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} tests passés")

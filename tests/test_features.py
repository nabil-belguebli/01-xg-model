"""Géométrie du tir. `pytest tests/` ou `python tests/test_features.py`."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import (  # noqa: E402
    _in_shot_triangle,
    assist_features,
    distance_to_goal,
    freeze_frame_features,
    shot_angle,
)


def test_penalty_spot_distance():
    """Point de penalty (108, 40) : 12 unités."""
    assert math.isclose(distance_to_goal(108, 40), 12.0)


def test_angle_decroit_avec_la_distance():
    """Dans l'axe, l'angle rétrécit avec la distance."""
    assert shot_angle(114, 40) > shot_angle(108, 40) > shot_angle(90, 40)


def test_angle_maximal_dans_laxe():
    """À distance égale, l'axe ouvre plus que l'aile."""
    assert shot_angle(108, 40) > shot_angle(108, 20)


def test_angle_quasi_nul_sur_la_ligne_de_sortie():
    """Depuis le coin, angle quasi nul."""
    assert shot_angle(120, 0) < 0.01


def test_angle_borne():
    """Entre 0 et pi, jamais NaN."""
    for x in range(1, 121, 7):
        for y in range(0, 81, 7):
            angle = shot_angle(float(x), float(y))
            assert 0.0 <= angle <= math.pi
            assert not math.isnan(angle)


def test_triangle_contient_defenseur_bien_place():
    """Défenseur entre le tireur et le but."""
    assert _in_shot_triangle(114.0, 40.0, sx=100.0, sy=40.0)


def test_triangle_exclut_defenseur_derriere_le_tireur():
    assert not _in_shot_triangle(90.0, 40.0, sx=100.0, sy=40.0)


def test_freeze_frame_compte_les_adversaires():
    shot = {"shot": {"freeze_frame": [
        {"location": [114.0, 40.0], "teammate": False,
         "position": {"name": "Center Back"}},
        {"location": [118.0, 40.0], "teammate": False,
         "position": {"name": "Goalkeeper"}},
        {"location": [114.0, 41.0], "teammate": True,
         "position": {"name": "Center Forward"}},
        {"location": [60.0, 10.0], "teammate": False,
         "position": {"name": "Left Back"}},
    ]}}
    out = freeze_frame_features(shot, x=100.0, y=40.0)

    assert out["defenders_in_triangle"] == 2  # gardien compté comme défenseur
    assert out["has_freeze_frame"]
    assert out["keeper_visible"]
    assert math.isclose(out["keeper_to_goal"], 2.0)  # gardien en (118, 40)
    assert math.isclose(out["keeper_to_shooter"], 18.0)
    assert math.isclose(out["nearest_defender"], 14.0)  # le coéquipier ne compte pas


def test_freeze_frame_absent():
    out = freeze_frame_features({"shot": {}}, x=100.0, y=40.0)
    assert out["defenders_in_triangle"] == 0
    assert not out["has_freeze_frame"]
    assert not out["keeper_visible"]
    assert math.isnan(out["keeper_to_goal"])
    assert math.isnan(out["nearest_defender"])


def test_sans_passe_decisive():
    assert assist_features(None) == {"assist_type": "Aucune", "assist_height": "Aucune"}


def test_type_de_passe_du_plus_specifique_au_moins_specifique():
    """Un centre en retrait est une passe en retrait ; un corner centré, un centre."""
    cut_back_cross = {"pass": {"cut_back": True, "cross": True, "height": {"name": "Ground Pass"}}}
    corner = {"pass": {"cross": True, "type": {"name": "Corner"}, "height": {"name": "High Pass"}}}
    short_corner = {"pass": {"type": {"name": "Corner"}, "height": {"name": "Ground Pass"}}}

    assert assist_features(cut_back_cross)["assist_type"] == "En retrait"
    assert assist_features(corner) == {"assist_type": "Centre", "assist_height": "High Pass"}
    assert assist_features(short_corner)["assist_type"] == "Coup de pied arrêté"


def test_passe_decisive_ignore_lissue_du_tir():
    """goal_assist n'existe que si le tir est un but : l'utiliser ferait fuiter la réponse."""
    with_goal = {"pass": {"goal_assist": True, "height": {"name": "Ground Pass"}}}
    without_goal = {"pass": {"shot_assist": True, "height": {"name": "Ground Pass"}}}
    assert assist_features(with_goal) == assist_features(without_goal)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} tests passés")

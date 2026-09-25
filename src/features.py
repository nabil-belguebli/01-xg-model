"""Tir StatsBomb -> vecteur de features.

Terrain 120 x 80. But en x=120, poteaux en y=36 et y=44.
"""

from __future__ import annotations

import math

GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
POST_LEFT_Y = 36.0
POST_RIGHT_Y = 44.0
GOAL_WIDTH = POST_RIGHT_Y - POST_LEFT_Y


def distance_to_goal(x: float, y: float) -> float:
    """Distance au centre de la ligne de but."""
    return math.hypot(GOAL_X - x, GOAL_CENTER_Y - y)


def shot_angle(x: float, y: float) -> float:
    """Angle (rad) sous lequel le tireur voit l'ouverture du but.

    Loi des cosinus sur le triangle tireur / poteau gauche / poteau droit.
    """
    a = math.hypot(GOAL_X - x, POST_LEFT_Y - y)
    b = math.hypot(GOAL_X - x, POST_RIGHT_Y - y)

    if a == 0.0 or b == 0.0:  # tir sur un poteau
        return 0.0

    cos_angle = (a**2 + b**2 - GOAL_WIDTH**2) / (2 * a * b)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # arrondi
    return math.acos(cos_angle)


def _in_shot_triangle(px: float, py: float, sx: float, sy: float) -> bool:
    """Point dans le triangle tireur / poteaux. Test des signes des produits vectoriels."""
    triangle = [(sx, sy), (GOAL_X, POST_LEFT_Y), (GOAL_X, POST_RIGHT_Y)]
    signs = []
    for i in range(3):
        ax, ay = triangle[i]
        bx, by = triangle[(i + 1) % 3]
        signs.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))

    return not (any(s < 0 for s in signs) and any(s > 0 for s in signs))


def freeze_frame_features(shot: dict, x: float, y: float) -> dict:
    """Adversaires dans le triangle de tir (gardien compris), gardien, défenseur le plus proche.

    Gardien absent du freeze frame : hors du champ de la caméra, donc loin de
    son but. Ses distances valent alors NaN, et keeper_visible le signale.
    """
    frame = shot.get("shot", {}).get("freeze_frame") or []
    opponents = [p for p in frame if not p.get("teammate", True)]

    defenders_in_triangle = sum(
        1 for p in opponents if _in_shot_triangle(p["location"][0], p["location"][1], x, y)
    )

    keeper = next((p for p in opponents if p.get("position", {}).get("name") == "Goalkeeper"), None)
    if keeper:
        kx, ky = keeper["location"][0], keeper["location"][1]
        keeper_to_goal = distance_to_goal(kx, ky)  # sorti de sa ligne ?
        keeper_to_shooter = math.hypot(kx - x, ky - y)  # face à face ?
    else:
        keeper_to_goal = keeper_to_shooter = math.nan

    field_players = [p for p in opponents if p is not keeper]
    nearest_defender = min(
        (math.hypot(p["location"][0] - x, p["location"][1] - y) for p in field_players),
        default=math.nan,
    )

    return {
        "defenders_in_triangle": defenders_in_triangle,
        "keeper_visible": keeper is not None,
        "keeper_to_goal": keeper_to_goal,
        "keeper_to_shooter": keeper_to_shooter,
        "nearest_defender": nearest_defender,
        "has_freeze_frame": len(frame) > 0,
    }


def assist_features(key_pass: dict | None) -> dict:
    """La passe qui a amené le tir. Aucune : récupération, dribble, coup franc direct...

    Un seul libellé par passe, du plus au moins spécifique. goal_assist et
    shot_assist sont ignorés : ils décrivent l'issue du tir, pas la passe.
    """
    if key_pass is None:
        return {"assist_type": "Aucune", "assist_height": "Aucune"}

    details = key_pass.get("pass", {})
    if details.get("through_ball"):
        assist_type = "Profondeur"
    elif details.get("cut_back"):
        assist_type = "En retrait"
    elif details.get("cross"):
        assist_type = "Centre"
    elif details.get("type", {}).get("name") in ("Corner", "Free Kick", "Throw-in"):
        assist_type = "Coup de pied arrêté"
    else:
        assist_type = "Autre passe"

    return {
        "assist_type": assist_type,
        "assist_height": details.get("height", {}).get("name", "Aucune"),
    }


def shot_to_row(shot: dict, match_id: int, key_pass: dict | None = None) -> dict:
    """Événement Shot -> une ligne de tableau.

    Seulement ce qui est connu au moment de la frappe : end_location,
    deflected, saved_to_post... décrivent l'issue et feraient fuiter la réponse.
    """
    x, y = shot["location"][0], shot["location"][1]
    details = shot.get("shot", {})

    row = {
        "match_id": match_id,
        "shot_id": shot["id"],
        "player": details.get("player", {}).get("name") or shot.get("player", {}).get("name"),
        "team": shot.get("team", {}).get("name"),
        "minute": shot.get("minute"),
        "x": x,
        "y": y,
        "distance": distance_to_goal(x, y),
        "angle": shot_angle(x, y),
        "body_part": details.get("body_part", {}).get("name"),
        "technique": details.get("technique", {}).get("name"),
        "shot_type": details.get("type", {}).get("name"),
        "under_pressure": bool(shot.get("under_pressure", False)),
        "first_time": bool(details.get("first_time", False)),
        "play_pattern": shot.get("play_pattern", {}).get("name"),
        "one_on_one": bool(details.get("one_on_one", False)),
        "open_goal": bool(details.get("open_goal", False)),
        "aerial_won": bool(details.get("aerial_won", False)),
        "statsbomb_xg": details.get("statsbomb_xg"),
        "is_goal": int(details.get("outcome", {}).get("name") == "Goal"),
    }
    row.update(freeze_frame_features(shot, x, y))
    row.update(assist_features(key_pass))
    return row

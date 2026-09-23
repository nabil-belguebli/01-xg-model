"""Téléchargement et cache des données ouvertes StatsBomb.

data/competitions.json, data/matches/{comp}/{season}.json, data/events/{match}.json
Source : https://github.com/statsbomb/open-data (à créditer).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = Path(__file__).resolve().parent.parent / "data" / "raw"


def _get(url: str, cache_path: Path, pause: float = 0.1) -> list | dict:
    if cache_path.exists():  # un fichier n'est téléchargé qu'une fois
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(pause)
    return payload


def competitions() -> list[dict]:
    return _get(f"{BASE}/competitions.json", CACHE / "competitions.json")


def matches(competition_id: int, season_id: int) -> list[dict]:
    return _get(
        f"{BASE}/matches/{competition_id}/{season_id}.json",
        CACHE / "matches" / str(competition_id) / f"{season_id}.json",
    )


def events(match_id: int) -> list[dict]:
    return _get(f"{BASE}/events/{match_id}.json", CACHE / "events" / f"{match_id}.json")


def shots(match_id: int) -> list[dict]:
    return [e for e in events(match_id) if e.get("type", {}).get("name") == "Shot"]

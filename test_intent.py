"""Test dello stadio NLP — solo la parte deterministica (regex + tabelle + vocabolario).

`route()` carica SetFit (torch): fuori dalla suite veloce. Qui si testano `parse_numbers`,
`bpm_from_speed`, la coerenza di `GOAL_PARAMS` e che le label siano ⊆ vocabolario di songs.csv.
"""
import csv
from pathlib import Path

from intent import (GOAL_LABELS, GOAL_PARAMS, GOAL_TO_EFFORT, MOOD_LABELS,
                    bpm_from_speed, parse_numbers)

CSV = Path(__file__).parent / "songs.csv"


def _csv_vocab(column: str) -> set[str]:
    values: set[str] = set()
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for tok in (row.get(column) or "").split(";"):
                tok = tok.strip()
                if tok:
                    values.add(tok)
    return values


def test_bpm_from_speed():
    assert bpm_from_speed(12) == 169     # 134 + 2.9*12 = 168.8 -> 169
    assert bpm_from_speed(3) == 150      # clamp minimo (cadenza naturale)
    assert bpm_from_speed(40) == 190     # clamp massimo


def test_parse_speed_kmh():
    assert parse_numbers("corro a 12 km/h")["speed_kmh"] == 12.0


def test_parse_pace_minkm_to_speed():
    # 5:00 min/km -> 12 km/h
    assert parse_numbers("tengo 5:00 min/km")["speed_kmh"] == 12.0


def test_distance_not_confused_with_speed():
    n = parse_numbers("oggi 10 km")
    assert n.get("distance_km") == 10.0 and "speed_kmh" not in n


def test_parse_duration():
    assert parse_numbers("30 minuti tranquilli")["duration_min"] == 30


def test_goal_params_consistent():
    assert set(GOAL_PARAMS) == set(GOAL_LABELS)
    for p in GOAL_PARAMS.values():
        assert 0.0 <= p["energy"] <= 1.0
        assert abs(p["w_bpm"] + p["w_mood"] - 1.0) < 1e-9   # pesi sommano a 1 (come AlgoRun)
        lo, hi = p["bpm"]
        assert lo < hi


def test_goal_to_effort_covers_all_goals():
    assert set(GOAL_TO_EFFORT) == set(GOAL_LABELS)


def test_labels_are_subset_of_csv_vocab():
    assert set(GOAL_LABELS) <= _csv_vocab("supports_goal")
    assert set(MOOD_LABELS) <= _csv_vocab("supports_mood")

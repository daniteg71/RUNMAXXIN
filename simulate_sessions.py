"""Genera due sessioni di allenamento simulate -> data/simulated/bpm_sessions.csv.

Formato atteso da build_dataset.py (una riga al secondo):
  session_id, user_id, second, bpm, resting_hr, max_hr, workout_goal, phase, speed_kmh, cadence_spm

Due casi opposti:
  A) 'marathon_ontarget'  — atleta da maratona: metriche precise, resta in zona target,
                             bpm stabile, NON si affatica (velocita' costante).
  B) 'push_then_fatigue'  — dice di voler spingere tantissimo, parte forte, poi si affatica:
                             il cuore sale (drift) ma la velocita' CROLLA -> va molto piu' lento.

Uso:  python simulate_sessions.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

OUT = Path(__file__).parent / "data" / "simulated" / "bpm_sessions.csv"
DURATION = 360   # secondi (6 minuti) per sessione

COLUMNS = ["session_id", "user_id", "second", "bpm", "resting_hr", "max_hr",
           "workout_goal", "phase", "speed_kmh", "cadence_spm"]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def marathon_ontarget() -> list[dict]:
    """Atleta preciso: riscaldamento poi zona target stabile, nessun affaticamento."""
    rows = []
    for s in range(DURATION):
        if s < 60:                                   # riscaldamento
            t = s / 60
            bpm, speed, cad, phase = lerp(120, 140, t), lerp(10, 15, t), lerp(165, 180, t), "warmup"
        else:                                        # zona target, costante
            bpm = 140 + 1.0 * math.sin(s / 15)       # micro-oscillazione, std bassissima
            speed, cad, phase = 15.0, 180.0, "run"
        rows.append({"session_id": "marathon_ontarget", "user_id": "athlete_A", "second": s,
                     "bpm": round(bpm, 1), "resting_hr": 50, "max_hr": 190,
                     "workout_goal": "ModerateRun", "phase": phase,
                     "speed_kmh": round(speed, 2), "cadence_spm": round(cad, 1)})
    return rows


def push_then_fatigue() -> list[dict]:
    """Parte fortissimo poi si affatica: HR sale (drift), velocita' crolla."""
    rows = []
    for s in range(DURATION):
        if s < 90:                                   # spinta iniziale
            t = s / 90
            bpm, speed, cad, phase = lerp(150, 180, t), lerp(14, 17, t), lerp(180, 188, t), "push"
        else:                                        # affaticamento
            t = (s - 90) / (DURATION - 90)
            bpm, speed, cad, phase = lerp(180, 190, t), lerp(17, 10, t), lerp(188, 158, t), "fatigue"
        rows.append({"session_id": "push_then_fatigue", "user_id": "runner_B", "second": s,
                     "bpm": round(bpm, 1), "resting_hr": 60, "max_hr": 195,
                     "workout_goal": "IntenseRun", "phase": phase,
                     "speed_kmh": round(speed, 2), "cadence_spm": round(cad, 1)})
    return rows


def main() -> None:
    rows = marathon_ontarget() + push_then_fatigue()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Scritte {len(rows)} righe in {OUT}")
    print("Sessioni: marathon_ontarget (athlete_A) + push_then_fatigue (runner_B)")


if __name__ == "__main__":
    main()

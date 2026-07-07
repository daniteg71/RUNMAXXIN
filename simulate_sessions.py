"""simulate_sessions.py — libreria di ARCHETIPI di prestazione (dati sensori simulati).

Genera data/simulated/bpm_sessions.csv con più sessioni, ognuna un tipo di corsa diverso —
durata diversa, andamento diverso, affaticamento diverso. Ogni riga = un secondo, con
battito, velocità e cadenza; l'input per `build_dataset.py`.

Servono al tester: scegli un PROMPT + uno di questi archetipi e guardi come cambia la musica.
Una corsa vera esportata da uno smartwatch è semplicemente un altro dataset in questo formato.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).parent / "data" / "simulated" / "bpm_sessions.csv"
random.seed(7)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def cadence_of(speed: float) -> float:
    return max(150.0, min(190.0, 150.0 + 2.0 * speed))


def build(session_id, user_id, resting, maxhr, goal, duration, fn) -> list[dict]:
    """fn(s, duration) -> (bpm, speed, phase). La cadenza deriva dalla velocità + rumore."""
    rows = []
    for s in range(duration):
        bpm, speed, phase = fn(s, duration)
        bpm += random.gauss(0, 1.3)
        speed = max(1.0, speed + random.gauss(0, 0.15))
        cad = max(120.0, cadence_of(speed) + random.gauss(0, 0.8))
        rows.append({"session_id": session_id, "user_id": user_id, "second": s,
                     "bpm": round(bpm, 2), "resting_hr": resting, "max_hr": maxhr,
                     "workout_goal": goal, "phase": phase,
                     "speed_kmh": round(speed, 2), "cadence_spm": round(cad, 2)})
    return rows


# ── ARCHETIPI (durate diverse, dinamiche diverse) ─────────────────────────────

def steady(s, dur):                                    # 35 min: ritmo costante in zona
    if s < 300:
        t = s / 300
        return lerp(120, 142, t), lerp(10, 15, t), "warmup"
    return 142, 15.0, "run"


def push_fatigue(s, dur):                              # 30 min: parte forte, cede
    if s < 480:
        t = s / 480
        return lerp(150, 180, t), lerp(14, 17, t), "push"
    t = (s - 480) / (dur - 480)
    return lerp(180, 193, t), lerp(17, 9, t), "fatigue"


def negative_split(s, dur):                            # 40 min: parte piano, accelera, controllato
    t = s / dur
    return lerp(128, 172, t), lerp(11, 16, t), "build"


def intervals(s, dur):                                 # 25 min: ripetute, oscilla
    if s < 300:
        t = s / 300
        return lerp(120, 150, t), lerp(9, 12, t), "warmup"
    cycle = (s - 300) % 300                             # 5 min per ciclo: 3 duro + 2 facile
    if cycle < 180:
        return 178, 17.0, "hard"
    return 146, 9.0, "easy"


def easy_recovery(s, dur):                             # 20 min: corsetta blanda, sforzo basso
    t = s / dur
    return lerp(112, 126, t), lerp(8.5, 9.5, t), "easy"


def beginner_struggle(s, dur):                         # 22 min: erratico, cuore alto, pause camminata
    if s < 180:
        t = s / 180
        return lerp(120, 176, t), lerp(8, 10, t), "spike"       # cuore schizza subito
    if (s // 120) % 2 == 1:                             # pause camminata ogni ~2 min
        return 168, 4.5, "walk"                         # rallenta ma il cuore resta alto
    return 182, 9.5, "struggle"


ARCHETYPES = [
    ("steady",            "athlete_A",  50, 190, "ModerateRun", 2100, steady),
    ("push_fatigue",      "runner_B",   60, 195, "IntenseRun",  1800, push_fatigue),
    ("negative_split",    "athlete_C",  52, 188, "ModerateRun", 2400, negative_split),
    ("intervals",         "runner_D",   55, 192, "IntenseRun",  1500, intervals),
    ("easy_recovery",     "runner_E",   58, 186, "EasyRun",     1200, easy_recovery),
    ("beginner_struggle", "runner_F",   70, 200, "EasyRun",     1320, beginner_struggle),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for sid, uid, rest, mx, goal, dur, fn in ARCHETYPES:
        all_rows.extend(build(sid, uid, rest, mx, goal, dur, fn))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Scritte {len(all_rows)} righe in {OUT}")
    print(f"Archetipi ({len(ARCHETYPES)}):")
    for sid, uid, rest, mx, goal, dur, _ in ARCHETYPES:
        print(f"  {sid:18s} {dur//60:2d} min   HR {rest}-{mx}   ({goal})")


if __name__ == "__main__":
    main()

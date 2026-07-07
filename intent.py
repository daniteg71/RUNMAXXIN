"""Stadio 1 (NLP) di RUNMAXXIN: frase -> {goal, mood, numbers, target_bpm, params}.

Stessa logica dell'estrattore d'intento di AlgoRun (regex per i numeri + SetFit per la
classificazione), ma le etichette sono riaddestrate sul vocabolario di `songs.csv`:
  - goal (tipo di allenamento): EasyRun / ModerateRun / IntenseRun   (colonna supports_goal)
  - mood:                       Neutral/Focused/Energetic/Motivated/Calm (colonna supports_mood)

Doppio regime (invariato rispetto ad AlgoRun):
  - QUANTITATIVO: se la frase dichiara una velocita'/passo, `target_bpm` e' il BPM
    "chirurgico" dalla cadenza (Van Dyck 2015, entrainment 1:1).
  - QUALITATIVO: se non ci sono numeri, comanda la banda BPM del tipo (`params`).

I numeri li estrae la regex (SetFit non estrae valori). Le bande/pesi/tau per tipo sono
scelte di design (banda ancorata alla cadenza naturale 150-190 spm, energia alla teoria
arousal-musica Karageorghis & Terry 2009) -> da giustificare con ablation nel paper.
"""
from __future__ import annotations

import re
from pathlib import Path

# --- modelli SetFit (lazy-load): uno per il goal, uno per il mood -------------
_GOAL_MODEL = None
_MOOD_MODEL = None
_GOAL_DIR = Path(__file__).parent / "models" / "intent-goal-setfit"
_MOOD_DIR = Path(__file__).parent / "models" / "intent-mood-setfit"

# --- vocabolario ancorato a songs.csv -----------------------------------------
GOAL_LABELS = ("EasyRun", "ModerateRun", "IntenseRun")
MOOD_LABELS = ("Neutral", "Focused", "Energetic", "Motivated", "Calm")

# ponte verso lo stadio 2/3: il tipo (testo) suggerisce la banda di sforzo; lo sforzo
# vero lo misura il sensore (Karvonen, effort_state in physiological_state.py).
GOAL_TO_EFFORT: dict[str, tuple[str, ...]] = {
    "EasyRun":     ("LowEffort", "TargetEffort"),
    "ModerateRun": ("TargetEffort",),
    "IntenseRun":  ("HighEffort", "VeryHighEffort"),
}

# numeri: velocità km/h, passo min/km, distanza km, durata min (regex, invariata da AlgoRun)
_SPEED = re.compile(r"(\d+(?:[.,]\d+)?)\s*km\s*/?\s*h", re.I)
_PACE = re.compile(r"(\d{1,2}):(\d{2})\s*(?:min)?\s*/?\s*km", re.I)
_DIST = re.compile(r"(\d+(?:[.,]\d+)?)\s*km(?!\s*/?\s*h)", re.I)
_DUR = re.compile(r"(\d+)\s*(?:min|minuti)\b", re.I)

# goal -> (banda BPM, energia target, pesi scoring, temperatura esplorazione).
# Ex TYPE_PARAMS di AlgoRun, ri-chiavata sulle 3 label del CSV (stessa struttura, stessa logica).
GOAL_PARAMS: dict[str, dict] = {
    "EasyRun":     {"bpm": (120, 135), "energy": 0.25, "w_bpm": 0.2, "w_mood": 0.8, "tau": 1.0},
    "ModerateRun": {"bpm": (140, 160), "energy": 0.55, "w_bpm": 0.5, "w_mood": 0.5, "tau": 0.4},
    "IntenseRun":  {"bpm": (165, 185), "energy": 0.90, "w_bpm": 0.85, "w_mood": 0.15, "tau": 0.15},
}


def _goal_model():
    global _GOAL_MODEL
    if _GOAL_MODEL is None:
        from setfit import SetFitModel
        _GOAL_MODEL = SetFitModel.from_pretrained(str(_GOAL_DIR))
    return _GOAL_MODEL


def _mood_model():
    global _MOOD_MODEL
    if _MOOD_MODEL is None:
        from setfit import SetFitModel
        _MOOD_MODEL = SetFitModel.from_pretrained(str(_MOOD_DIR))
    return _MOOD_MODEL


def predict_goal(text: str) -> str:
    """Frase -> tipo di allenamento (EasyRun/ModerateRun/IntenseRun)."""
    return str(_goal_model().predict([text])[0])


def predict_mood(text: str) -> str:
    """Frase -> mood (Neutral/Focused/Energetic/Motivated/Calm)."""
    return str(_mood_model().predict([text])[0])


def parse_numbers(text: str) -> dict:
    """Estrae i valori quantitativi dalla frase (invariato da AlgoRun)."""
    n: dict = {}
    if (m := _SPEED.search(text)):
        n["speed_kmh"] = float(m.group(1).replace(",", "."))
    elif (m := _PACE.search(text)):
        pace = int(m.group(1)) + int(m.group(2)) / 60
        n["speed_kmh"] = round(60 / pace, 1) if pace else None
    if (m := _DIST.search(text)):
        n["distance_km"] = float(m.group(1).replace(",", "."))
    if (m := _DUR.search(text)):
        n["duration_min"] = int(m.group(1))
    return n


# velocità dichiarata -> cadenza (spm) -> BPM target. BPM = cadenza (entrainment 1:1,
# Van Dyck 2015). Regressione clampata alla cadenza naturale di corsa 150-190. (invariato)
_CAD_INTERCEPT, _CAD_SLOPE, _CAD_MIN, _CAD_MAX = 134.0, 2.9, 150.0, 190.0


def bpm_from_speed(speed_kmh: float) -> int:
    """Velocità (km/h) -> BPM desiderato (calcolo 'chirurgico', regime quantitativo)."""
    return round(min(_CAD_MAX, max(_CAD_MIN, _CAD_INTERCEPT + _CAD_SLOPE * speed_kmh)))


def route(text: str) -> dict:
    """Frase -> {goal, mood, numbers, target_bpm, params}.

    Doppio regime (identico ad AlgoRun): se la velocità è dichiarata (quantitativo),
    `target_bpm` è il BPM 'chirurgico' dalla cadenza; altrimenti None e comanda la banda
    di `params` (qualitativo). `goal` = tipo di allenamento riconosciuto sempre.
    """
    goal = predict_goal(text)
    mood = predict_mood(text)
    numbers = parse_numbers(text)
    target_bpm = bpm_from_speed(numbers["speed_kmh"]) if numbers.get("speed_kmh") else None
    return {"goal": goal, "mood": mood, "numbers": numbers,
            "target_bpm": target_bpm, "params": GOAL_PARAMS[goal]}


if __name__ == "__main__":
    import sys
    print(route(" ".join(sys.argv[1:]) or "oggi ripetute veloci a 12 km/h, sono carico"))

#  the following code represents the first stage of the pipeline and contains
#  the NLP module that transforms a user sentence into structured data

#  the overall code works as follows:


#   user sentence
#           ↓
#   workout type
#   mood
#   numerical values
#   target BPM
#   regime parameters



#  the following libraries are used:
#    - annotations to use more flexible type annotations
#    - re for regular expressions, used to extract numbers from the sentence
#    - Path to build the paths of the folders containing the models

from __future__ import annotations
import re
from pathlib import Path



#  two separate models are used:
#    - goal model, to predict the workout type (EasyRun/ModerateRun/IntenseRun)
#    - mood model, to predict the mood (Neutral/Focused/Energetic/Motivated/Calm)

_GOAL_MODEL = None
_MOOD_MODEL = None
_GOAL_DIR = Path(__file__).parent / "models" / "intent-goal-setfit"
_MOOD_DIR = Path(__file__).parent / "models" / "intent-mood-setfit"



#  labels used to train the SetFit model, with vocabulary aligned to songs.csv

GOAL_LABELS = ("EasyRun", "ModerateRun", "IntenseRun")
MOOD_LABELS = ("Neutral", "Focused", "Energetic", "Motivated", "Calm")


#  dictionary that maps workout type → effort detected by the sensor
#  bridge between linguistic intention and physiological state

GOAL_TO_EFFORT: dict[str, tuple[str, ...]] = {
    "EasyRun":     ("LowEffort", "TargetEffort"),
    "ModerateRun": ("TargetEffort",),
    "IntenseRun":  ("HighEffort", "VeryHighEffort"),
}


#  before querying the SetFit model, the code tries to identify the
#  workout type through specific keywords

GOAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "IntenseRun": ("ripetut", "intervall", "scatt", "sprint", "spinger", "spingo",
                   "a tutta", "massimo sforzo", "al limite", "distrugg", "ammazz",
                   "sfinir", "sfianc", "fartlek", "soglia"),
    "EasyRun":    ("recupero", "defatic", "rigenera", "scioglier", "blando",
                   "molto piano", "passeggiat"),
    "ModerateRun": ("maratona", "fondo", "medio", "ritmo costante", "endurance", "lungo lento"),
}


#  the following function converts the sentence to lowercase and checks
#  whether any keyword is present in the user's sentence

def goal_from_keywords(text: str) -> str | None:
    low = text.lower()
    for goal, words in GOAL_KEYWORDS.items():
        if any(w in low for w in words):
            return goal
    return None


#  regular expressions are used to detect values (speed/pace/distance/duration)

_SPEED = re.compile(r"(\d+(?:[.,]\d+)?)\s*km\s*/?\s*h", re.I)
_PACE = re.compile(r"(\d{1,2}):(\d{2})\s*(?:min)?\s*/?\s*km", re.I)
_DIST = re.compile(r"(\d+(?:[.,]\d+)?)\s*km(?!\s*/?\s*h)", re.I)
_DUR = re.compile(r"(\d+)\s*(?:min|minuti)\b", re.I)


#  this dictionary connects the NLP output with the controller and recommender

GOAL_PARAMS: dict[str, dict] = {
    "EasyRun":     {"bpm": (120, 135), "energy": 0.25, "w_bpm": 0.2, "w_mood": 0.8, "tau": 1.0},
    "ModerateRun": {"bpm": (140, 160), "energy": 0.55, "w_bpm": 0.5, "w_mood": 0.5, "tau": 0.4},
    "IntenseRun":  {"bpm": (165, 185), "energy": 0.90, "w_bpm": 0.85, "w_mood": 0.15, "tau": 0.15},
}


#  the following two functions load the models only when they are used for the first time

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


#  in the following two functions, the sentence text is placed inside a list
#  and the model predicts the goal/mood.
#  the first prediction [0] is extracted and cast to a string

def predict_goal(text: str) -> str:
    return str(_goal_model().predict([text])[0])


def predict_mood(text: str) -> str:
    return str(_mood_model().predict([text])[0])



#  the following function extracts the values (speed/pace/distance/duration)
#  and then stores them inside a dictionary

def parse_numbers(text: str) -> dict:
    n: dict = {}
    if (m := _SPEED.search(text)):
        n["speed_kmh"] = float(m.group(1).replace(",", "."))
    elif (m := _PACE.search(text)):
        pace = int(m.group(1)) + int(m.group(2)) / 60
        # speed = 60 / pace
        n["speed_kmh"] = round(60 / pace, 1) if pace else None      
    if (m := _DIST.search(text)):
        n["distance_km"] = float(m.group(1).replace(",", "."))
    if (m := _DUR.search(text)):
        n["duration_min"] = int(m.group(1))
    return n


#  the following parameters are derived from the 2015 Van Dyck paper.
#  they are calculated through linear regression (x = speed, y = cadence)

_CAD_INTERCEPT, _CAD_SLOPE, _CAD_MIN, _CAD_MAX = 134.0, 2.9, 150.0, 190.0


#  the following function derives the target BPM from speed 

def bpm_from_speed(speed_kmh: float) -> int:
    return round(min(_CAD_MAX, max(_CAD_MIN, _CAD_INTERCEPT + _CAD_SLOPE * speed_kmh)))


#  main function that coordinates all operations

def route(text: str) -> dict:

    # extracts the goal: it first checks the keyword dictionary, otherwise it applies SetFit
    goal = goal_from_keywords(text) or predict_goal(text)
    
    # extracts the mood through SetFit
    mood = predict_mood(text)

    # extracts (speed_kmh, distance_km, duration_min)
    numbers = parse_numbers(text)

    # if speed_kmh is available, the target BPM is derived
    target_bpm = bpm_from_speed(numbers["speed_kmh"]) if numbers.get("speed_kmh") else None
    
    return {"goal": goal, "mood": mood, "numbers": numbers,
            "target_bpm": target_bpm, "params": GOAL_PARAMS[goal]}   # parameters are retrieved according to the goal
                                                                     # (BPM range, energy, weights, tau)




# simulation 

if __name__ == "__main__":
    import sys
    print(route(" ".join(sys.argv[1:]) or "oggi ripetute veloci a 12 km/h, sono carico"))

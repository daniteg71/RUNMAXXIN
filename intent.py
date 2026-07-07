#  il seguente codice rappresenta il primo stadio della pipeline, contiene
#  il modulo NLP che trasforma una frase dell'utente in dati strutturati

#  il codice nel complesso lavora in questo modo:


#   frase dell’utente
#           ↓
#   tipo di allenamento
#   mood
#   valori numerici
#   BPM target
#   parametri del regime



#  vengono utilizzate le librerie:
#    - annotations per usare annotazioni di tipo più flessibili
#    - re per le espressioni regolari, ovvero per estrarre i numeri dalla frase
#    - path per costruire i percorsi delle cartelle contenenti i modelli

from __future__ import annotations
import re
from pathlib import Path



#  vengono utilizzati modelli distinti:
#    - goal model, per prevedere il tipo di allenamento (EasyRun/ModerateRun/IntenseRun)
#    - mood model, per prevdere il mood (Neutral/Focused/Energetic/Motivated/Calm)

_GOAL_MODEL = None
_MOOD_MODEL = None
_GOAL_DIR = Path(__file__).parent / "models" / "intent-goal-setfit"
_MOOD_DIR = Path(__file__).parent / "models" / "intent-mood-setfit"



#  label utilizzate per allenare modello SetFit, vocabolario ancorato a songs.csv

GOAL_LABELS = ("EasyRun", "ModerateRun", "IntenseRun")
MOOD_LABELS = ("Neutral", "Focused", "Energetic", "Motivated", "Calm")


#  dizionario che mappa tipo di allenamento → effort rilevato dal sensore
#  ponte tra intenzione linguistica e stato fisiologico

GOAL_TO_EFFORT: dict[str, tuple[str, ...]] = {
    "EasyRun":     ("LowEffort", "TargetEffort"),
    "ModerateRun": ("TargetEffort",),
    "IntenseRun":  ("HighEffort", "VeryHighEffort"),
}


#  prima di interrogare il modello SetFit, si prova a riconoscere il 
#  tipo di allenamento attraverso delle specifiche keywords

GOAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "IntenseRun": ("ripetut", "intervall", "scatt", "sprint", "spinger", "spingo",
                   "a tutta", "massimo sforzo", "al limite", "distrugg", "ammazz",
                   "sfinir", "sfianc", "fartlek", "soglia"),
    "EasyRun":    ("recupero", "defatic", "rigenera", "scioglier", "blando",
                   "molto piano", "passeggiat"),
    "ModerateRun": ("maratona", "fondo", "medio", "ritmo costante", "endurance", "lungo lento"),
}


#  la seguente funzione converte la frase in minuscolo e controlla 
#  la presenza di keyword all'interno della frase dell'utente

def goal_from_keywords(text: str) -> str | None:
    low = text.lower()
    for goal, words in GOAL_KEYWORDS.items():
        if any(w in low for w in words):
            return goal
    return None


#  viene utilizzato una Regex per riconoscere i valori (velocità/passo/distanza/durata)

_SPEED = re.compile(r"(\d+(?:[.,]\d+)?)\s*km\s*/?\s*h", re.I)
_PACE = re.compile(r"(\d{1,2}):(\d{2})\s*(?:min)?\s*/?\s*km", re.I)
_DIST = re.compile(r"(\d+(?:[.,]\d+)?)\s*km(?!\s*/?\s*h)", re.I)
_DUR = re.compile(r"(\d+)\s*(?:min|minuti)\b", re.I)


#  questo dizionario rappresenta il collegamento fra l'output NLP ed il controller e reccomender

GOAL_PARAMS: dict[str, dict] = {
    "EasyRun":     {"bpm": (120, 135), "energy": 0.25, "w_bpm": 0.2, "w_mood": 0.8, "tau": 1.0},
    "ModerateRun": {"bpm": (140, 160), "energy": 0.55, "w_bpm": 0.5, "w_mood": 0.5, "tau": 0.4},
    "IntenseRun":  {"bpm": (165, 185), "energy": 0.90, "w_bpm": 0.85, "w_mood": 0.15, "tau": 0.15},
}


#  le seguenti 2 funzioni caricano i modelli solamente quando vengono usati per la prima volta

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


#  nelle seguenti due funzioni il testo della frase viene inserito in una lista
#  ed il modello predice goal/mood. 
#  viene estratta la prima predizione [0] e la si casta a stringa

def predict_goal(text: str) -> str:
    return str(_goal_model().predict([text])[0])


def predict_mood(text: str) -> str:
    return str(_mood_model().predict([text])[0])



#  la seguente funzione estrae i valori (velocità/passo/distanza/durata)
#  per poi salvarli all'interno di un dizionario

def parse_numbers(text: str) -> dict:
    n: dict = {}
    if (m := _SPEED.search(text)):
        n["speed_kmh"] = float(m.group(1).replace(",", "."))
    elif (m := _PACE.search(text)):
        pace = int(m.group(1)) + int(m.group(2)) / 60
        #velocità=pace/60
        n["speed_kmh"] = round(60 / pace, 1) if pace else None      
    if (m := _DIST.search(text)):
        n["distance_km"] = float(m.group(1).replace(",", "."))
    if (m := _DUR.search(text)):
        n["duration_min"] = int(m.group(1))
    return n


#  i seguenti parametri sono ricavati dal paper di Van Dyck del 2015.
#  calcolati attraverso una regressione lineare (x=velocità, y=cadenza)

_CAD_INTERCEPT, _CAD_SLOPE, _CAD_MIN, _CAD_MAX = 134.0, 2.9, 150.0, 190.0


#  la seguente funzione ricava dalla velocità i BPM target 

def bpm_from_speed(speed_kmh: float) -> int:
    return round(min(_CAD_MAX, max(_CAD_MIN, _CAD_INTERCEPT + _CAD_SLOPE * speed_kmh)))


#  funzione principale che coordina tutte le operazioni

def route(text: str) -> dict:

    #estrae goal, prima prova a vedere se è presente nel dizionario, sennò applica SetFit
    goal = goal_from_keywords(text) or predict_goal(text)
    
    #estrae mood tramite SetFit
    mood = predict_mood(text)

    #estrae (speed_kmh, distance_km, duration_min)
    numbers = parse_numbers(text)

    #se esiste speed_kmh si ricava i BPM target
    target_bpm = bpm_from_speed(numbers["speed_kmh"]) if numbers.get("speed_kmh") else None
    
    return {"goal": goal, "mood": mood, "numbers": numbers,
            "target_bpm": target_bpm, "params": GOAL_PARAMS[goal]}   #in base al goal vengono recuperati parametri
                                                                     #(banda BPM, energy, pesi, tau)




# simulazione 

if __name__ == "__main__":
    import sys
    print(route(" ".join(sys.argv[1:]) or "oggi ripetute veloci a 12 km/h, sono carico"))

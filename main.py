"""main.py — avvio di RUNMAXXIN: prompt -> intento (NLP) -> vettore target (controller).

Il recommender (collega) sceglie la canzone dal vettore target. Qui ci si ferma al target.

Uso:  python main.py "oggi ripetute a 12 km/h, sono carico"
"""
import sys

from intent import route
from controller import decide

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "oggi ripetute a 12 km/h, sono carico"
    intent = route(prompt)          # STADIO 1 · NLP
    target = decide(intent)         # CONTROLLER -> vettore
    print(f"PROMPT : {prompt}")
    print(f"INTENT : goal={intent['goal']} mood={intent['mood']} target_bpm={intent['target_bpm']}")
    print(f"TARGET : vettore={target.as_vector()} (bpm,energy,valence) "
          f"regime={target.regime} tau={target.tau} tol=±{target.bpm_tolerance}")
    # -> il recommender del collega sceglie la canzone piu' vicina a target.as_vector()

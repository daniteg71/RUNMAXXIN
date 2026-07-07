"""main.py — avvio di RUNMAXXIN (cold start): prompt -> intento -> Target -> canzoni.

  prompt --NLP--> intent --controller--> Target --recommender(collega)--> Top-K canzoni

Uso:  python main.py "oggi ripetute a 12 km/h, sono carico"
"""
import sys

from intent import route
from controller import decide
import recommender

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "oggi ripetute a 12 km/h, sono carico"
    intent = route(prompt)          # STADIO 1 · NLP
    target = decide(intent)         # CONTROLLER -> vettore (cold start)
    print(f"PROMPT : {prompt}")
    print(f"INTENT : goal={intent['goal']} mood={intent['mood']} target_bpm={intent['target_bpm']}")
    print(f"TARGET : vettore={target.as_vector()} regime={target.regime} tau={target.tau}\n")
    top = recommender.recommend(target, top_k=3)          # STADIO 3 · recommender del collega
    print(top[["title", "artist", "genre", "bpm", "probability_percent"]].to_string(index=False))

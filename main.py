"""main.py — orchestra i 3 stadi di RUNMAXXIN (minimale).

  prompt --NLP--> intent --sensori--> controller.decide --> Target --> recommend --> canzone

Il recommender vero lo fa il collega: qui c'e' uno STUB (bpm piu' vicino) facilmente
sostituibile con distance + softmax sul vettore Target.

Uso:
  python main.py "oggi ripetute a 12 km/h, sono carico"
  python main.py "corsa tranquilla di recupero" --windows data/processed/physiological_windows.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from statistics import mean

from intent import route
from controller import decide


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recommend(target, songs: list[dict], exclude: set) -> dict:
    """STUB del recommender (lo fa il collega): la canzone col bpm piu' vicino al target,
    ristretta ai generi del mood se presenti. Sostituibile con distance+softmax sul vettore."""
    cands = [s for s in songs if s.get("song_id") not in exclude]
    if target.genres:
        filt = [s for s in cands if s.get("genre") in target.genres]
        cands = filt or cands
    return min(cands, key=lambda s: abs(float(s["bpm"]) - target.bpm))


def aggregate(chunk: list[dict]) -> dict:
    """Un gruppo di finestre -> uno 'stato sensori' medio per la canzone."""
    return {
        "mean_hrr": mean(float(w["mean_hrr"]) for w in chunk),
        "effort_state": Counter(w["effort_state"] for w in chunk).most_common(1)[0][0],
        "trend_state": Counter(w["trend_state"] for w in chunk).most_common(1)[0][0],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RUNMAXXIN — prompt -> canzoni per allenarsi.")
    ap.add_argument("prompt", nargs="+", help="il prompt dell'utente (<=20 parole)")
    ap.add_argument("--songs", default="songs.csv")
    ap.add_argument("--windows", default=None, help="CSV finestre da build_dataset (opzionale)")
    ap.add_argument("--per", type=int, default=6, help="finestre sensori per canzone")
    args = ap.parse_args()
    prompt = " ".join(args.prompt)

    intent = route(prompt)                       # STADIO 1 · NLP
    print(f"PROMPT: {prompt}")
    print(f"INTENT: goal={intent['goal']} mood={intent['mood']} "
          f"target_bpm={intent['target_bpm']} numeri={intent['numbers']}\n")

    songs = load_csv(args.songs)
    played: set = set()
    last_bpm = None

    # Nessun sensore -> cold start: un solo target/canzone.
    if not args.windows:
        t = decide(intent)                       # STADIO 2 assente + CONTROLLER
        song = recommend(t, songs, played)       # STADIO 3 (stub collega)
        print(f"[cold start] target {t.bpm} bpm energy={t.energy} regime={t.regime} "
              f"-> {song['title']} — {song['artist']} ({song['bpm']} bpm, {song['genre']})")
        return

    # Con sensori: loop sulle finestre, una canzone ogni --per finestre.
    windows = load_csv(args.windows)
    print(f"Sessione: {len(windows)} finestre, {args.per} per canzone\n")
    for i in range(0, len(windows), args.per):
        chunk = windows[i:i + args.per]
        if not chunk:
            break
        state = aggregate(chunk)                 # STADIO 2 · sensori
        t = decide(intent, state, last_bpm)      # CONTROLLER -> vettore target
        song = recommend(t, songs, played)       # STADIO 3 (stub collega)
        played.add(song.get("song_id"))
        last_bpm = float(song["bpm"])
        tag = "RECUPERO" if t.recovery else t.regime
        print(f"HRR {state['mean_hrr']:.2f} {state['effort_state']:<14} "
              f"target {t.bpm:>5} bpm [{tag}] -> {song['title']} ({song['bpm']} bpm, {song['genre']})")


if __name__ == "__main__":
    main()

"""session.py — loop di sessione di RUNMAXXIN.

Concatena gli stadi SENZA toccare i file base: legge le finestre prodotte dal `build_dataset.py`
(physiological_windows.csv), una finestra ogni 30s, le raggruppa per canzone, e per ogni canzone:
controller.decide -> Target -> recommender.recommend (il modulo del collega).

  intent (una volta) ─┐
                      ├─► decide(intent, stato_sensori, last_bpm, elapsed) ─► Target
  finestre 30s ───────┘                                                        │
                                                     recommender.recommend(Target) ─► canzone

Uso:
  python build_dataset.py --input data/simulated/bpm_sessions.csv --output data/processed/physiological_windows.csv
  python session.py "oggi voglio spingere forte" --session push_then_fatigue
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from statistics import mean

from intent import route
from controller import decide
import recommender


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_song(windows: list[dict], song_seconds: int) -> list[list[dict]]:
    """Raggruppa le finestre da 30s (dal build_dataset) in blocchi da song_seconds = 1 canzone."""
    blocks: dict[int, list[dict]] = defaultdict(list)
    for w in windows:
        blocks[int(w["window_start_second"]) // song_seconds].append(w)
    return [blocks[k] for k in sorted(blocks)]


def aggregate(block: list[dict]) -> dict:
    """Le finestre del blocco -> uno stato sensori per la canzone (media HRR, effort/trend dominanti)."""
    return {"mean_hrr": mean(float(w["mean_hrr"]) for w in block),
            "effort_state": Counter(w["effort_state"] for w in block).most_common(1)[0][0],
            "trend_state": Counter(w["trend_state"] for w in block).most_common(1)[0][0]}


def run(prompt: str, windows: list[dict], song_seconds: int) -> None:
    intent = route(prompt)                                     # STADIO 1 · NLP (una volta)
    print(f"PROMPT : {prompt}")
    print(f"INTENT : goal={intent['goal']} mood={intent['mood']} target_bpm={intent['target_bpm']}\n")
    played: list[str] = []
    last_bpm = None
    for block in group_by_song(windows, song_seconds):
        state = aggregate(block)                               # STADIO 2 · finestre 30s del build_dataset
        elapsed = int(block[0]["window_start_second"]) / 60.0
        target = decide(intent, state, last_bpm, elapsed_min=elapsed)   # CONTROLLER -> Target
        top = recommender.recommend(target, top_k=1, exclude_song_ids=played)   # STADIO 3 · collega
        song = top.iloc[0]
        played.append(str(song["song_id"]))
        last_bpm = float(song["bpm"])
        tag = "RECUPERO" if target.recovery else target.regime
        print(f"  t={elapsed:4.1f}m HRR {state['mean_hrr']:.2f} {state['effort_state']:<14} "
              f"target {target.bpm:>5}bpm [{tag:11}] -> {str(song['title'])[:30]:30} "
              f"({song['bpm']}bpm {song['genre']}, {song['probability_percent']:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Loop di sessione RUNMAXXIN.")
    ap.add_argument("prompt", nargs="+", help="il prompt dell'utente (<=20 parole)")
    ap.add_argument("--windows", default="data/processed/physiological_windows.csv",
                    help="output di build_dataset.py (finestre 30s)")
    ap.add_argument("--session", default=None, help="filtra su una session_id")
    ap.add_argument("--song-seconds", type=int, default=120, help="durata blocco-canzone")
    a = ap.parse_args()
    windows = load(a.windows)
    if a.session:
        windows = [w for w in windows if w["session_id"] == a.session]
    run(" ".join(a.prompt), windows, a.song_seconds)


if __name__ == "__main__":
    main()

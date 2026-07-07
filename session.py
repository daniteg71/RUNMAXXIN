"""session.py — loop di sessione di RUNMAXXIN.

Concatena gli stadi SENZA toccare i file base: legge le finestre prodotte dal TUO
`build_dataset.py` (physiological_windows.csv), una finestra ogni 30s, le raggruppa per
canzone, e per ogni canzone: controller.decide -> Target -> recommend (gancio del collega).

  intent (una volta) ─┐
                      ├─► decide(intent, stato_sensori, last_bpm, elapsed) ─► Target ─► recommend
  finestre 30s ───────┘        (aggregate delle finestre del blocco-canzone)

Il recommender e' un gancio: qui c'e' uno STUB (bpm piu' vicino nei generi del mood),
il collega lo sostituisce con distanza pesata + softmax su target.as_vector().

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


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recommend(target, songs: list[dict], exclude: set) -> dict:
    """GANCIO del collega. Stub: bpm piu' vicino, ristretto ai generi del mood.
    Sostituire con distanza pesata + softmax su target.as_vector()."""
    c = [s for s in songs if s.get("song_id") not in exclude]
    if target.genres:
        f = [s for s in c if s.get("genre") in target.genres]
        c = f or c
    return min(c, key=lambda s: abs(float(s["bpm"]) - target.bpm))


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


def run(prompt: str, windows: list[dict], songs: list[dict], song_seconds: int) -> None:
    intent = route(prompt)                                     # STADIO 1 · NLP (una volta)
    print(f"PROMPT : {prompt}")
    print(f"INTENT : goal={intent['goal']} mood={intent['mood']} target_bpm={intent['target_bpm']}\n")
    played: set = set()
    last_bpm = None
    for block in group_by_song(windows, song_seconds):
        state = aggregate(block)                               # STADIO 2 · finestre 30s del build_dataset
        elapsed = int(block[0]["window_start_second"]) / 60.0
        target = decide(intent, state, last_bpm, elapsed_min=elapsed)   # CONTROLLER -> Target
        song = recommend(target, songs, played)               # STADIO 3 · gancio collega
        played.add(song.get("song_id"))
        last_bpm = float(song["bpm"])
        tag = "RECUPERO" if target.recovery else target.regime
        print(f"  t={elapsed:4.1f}m HRR {state['mean_hrr']:.2f} {state['effort_state']:<14} "
              f"target {target.bpm:>5}bpm [{tag:11}] -> {song['title'][:30]:30} ({song['bpm']}bpm {song['genre']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Loop di sessione RUNMAXXIN.")
    ap.add_argument("prompt", nargs="+", help="il prompt dell'utente (<=20 parole)")
    ap.add_argument("--windows", default="data/processed/physiological_windows.csv",
                    help="output di build_dataset.py (finestre 30s)")
    ap.add_argument("--songs", default="songs.csv")
    ap.add_argument("--session", default=None, help="filtra su una session_id")
    ap.add_argument("--song-seconds", type=int, default=120, help="durata blocco-canzone")
    a = ap.parse_args()
    windows = load(a.windows)
    if a.session:
        windows = [w for w in windows if w["session_id"] == a.session]
    run(" ".join(a.prompt), windows, load(a.songs), a.song_seconds)


if __name__ == "__main__":
    main()

"""session.py — loop di sessione di RUNMAXXIN.

Concatena gli stadi SENZA toccare i file base: legge le finestre prodotte dal `build_dataset.py`
(physiological_windows.csv), una finestra ogni 30s, le raggruppa per canzone, e per ogni canzone:
controller.decide -> Target -> recommender.recommend (il modulo del collega) -> effort-gate.

  intent (una volta) ─┐
                      ├─► decide(intent, stato_sensori, last_bpm, elapsed) ─► Target
  finestre 30s ───────┘                                                        │
                                    recommender.recommend(Target, top_k=K) ─► Top-K candidati
                                                                                 │
                              symbolic.is_effort_compatible (Generator->Validator) ─► canzone

Il recommender ottimizza la distanza vettoriale [bpm,energy,valence] e può ignorare la
compatibilità di sforzo (misurato: 16.7% di violazioni su un campione ampio di casi). Il
gate simbolico scorre il Top-K e sceglie la prima canzone compatibile con effort_band.

Uso:
  python build_dataset.py --input data/simulated/bpm_sessions.csv --output data/processed/physiological_windows.csv
  python session.py "oggi voglio spingere forte" --session push_then_fatigue
"""
from __future__ import annotations

import argparse
import csv
import random
import zlib
from collections import Counter, defaultdict
from statistics import mean

from intent import route
from controller import decide
from symbolic import is_effort_compatible
import recommender

TOP_K = 5            # candidati per il gate (moltiplicato per il pool di campionamento)
SAMPLE_POOL = 10     # fra quante candidate vicine campionare (varietà, restando vicino al target)


def load_effort_by_song(path: str = "songs.csv") -> dict[str, str]:
    """song_id -> matches_effort. recommender.recommend() non la restituisce
    nell'output (colonne fisse); la si recupera qui senza toccare il suo modulo."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["song_id"]: r.get("matches_effort", "") for r in csv.DictReader(f)}


def load_song_variants(path: str = "songs.csv") -> dict:
    """(title, artist) -> tutti i song_id che sono la STESSA canzone. Il catalogo Spotify di
    base ha fino a 45 copie della stessa traccia con id diversi; serve a non riproporre un
    doppione (l'esclusione per song_id da sola non basta)."""
    variants: dict = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            variants[(r["title"], r["artist"])].append(str(r["song_id"]))
    return variants


def pick_song(rng, target, top_df, effort_by_song: dict[str, str]):
    """Sceglie la canzone: deduplica per (titolo, artista), filtra con l'effort gate, poi
    CAMPIONA fra le vicine pesando per `probability`.

    Il recommender del collega calcola il softmax (con tau) ma poi ordina e prende i primi:
    è deterministico, quindi target simili -> stesse canzoni sempre. Qui completiamo il softmax
    campionando (Boltzmann, Sutton & Barto): sessioni diverse -> canzoni diverse; con un `rng`
    seminato per sessione resta riproducibile. Ritorna (canzone, esplorato, top-list dedup)."""
    seen, rows = set(), []
    for _, r in top_df.iterrows():
        k = (r["title"], r["artist"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    compat = [r for r in rows if is_effort_compatible(
        effort_by_song.get(str(r["song_id"]), "").split(";"), target.effort_band)]
    pool = (compat or rows)[:SAMPLE_POOL]
    weights = [max(1e-12, float(r["probability"])) for r in pool]
    chosen = rng.choices(pool, weights=weights, k=1)[0]
    explored = bool(rows) and str(chosen["song_id"]) != str(rows[0]["song_id"])
    return chosen, explored, rows


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
    """Le finestre del blocco -> uno stato sensori per la canzone (media HRR, effort/trend
    dominanti, e velocità/cadenza medie: servono al controller per far seguire i BPM alla
    velocità reale — live_entrainment_bpm)."""
    def avg(col):
        vals = [float(w[col]) for w in block if w.get(col) not in (None, "")]
        return mean(vals) if vals else None
    return {"mean_hrr": mean(float(w["mean_hrr"]) for w in block),
            "effort_state": Counter(w["effort_state"] for w in block).most_common(1)[0][0],
            "trend_state": Counter(w["trend_state"] for w in block).most_common(1)[0][0],
            "mean_speed_kmh": avg("mean_speed_kmh"),
            "mean_cadence_spm": avg("mean_cadence_spm")}


def run(prompt: str, windows: list[dict], song_seconds: int, session_id: str = "") -> None:
    intent = route(prompt)                                     # STADIO 1 · NLP (una volta)
    print(f"PROMPT : {prompt}")
    print(f"INTENT : goal={intent['goal']} mood={intent['mood']} target_bpm={intent['target_bpm']}\n")
    effort_by_song = load_effort_by_song()
    variants = load_song_variants()
    rng = random.Random(zlib.crc32((session_id or prompt).encode()))   # riproducibile, vario tra sessioni
    played: list[str] = []
    last_bpm = None
    for block in group_by_song(windows, song_seconds):
        state = aggregate(block)                               # STADIO 2 · finestre 30s del build_dataset
        elapsed = int(block[0]["window_start_second"]) / 60.0
        target = decide(intent, state, last_bpm, elapsed_min=elapsed)   # CONTROLLER -> Target
        top = recommender.recommend(target, top_k=TOP_K * 8, exclude_song_ids=played)   # STADIO 3 · collega
        song, _explored, _rows = pick_song(rng, target, top, effort_by_song)   # gate + softmax sampling
        played.extend(variants.get((song["title"], song["artist"]), [str(song["song_id"])]))
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
    run(" ".join(a.prompt), windows, a.song_seconds, session_id=a.session or "")


if __name__ == "__main__":
    main()

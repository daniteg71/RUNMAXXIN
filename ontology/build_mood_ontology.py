"""Genera l'ontologia mood -> generi musicali (ontology/genre_mood.ttl).

Logica (ancorata al dataset, NON inventata): per ognuno dei 5 mood si associano i generi
del catalogo `songs.csv` guardando la colonna `supports_mood`. Per ogni genere si calcola la
quota di brani che supportano ciascun mood; il genere e' associato (`ar:genreSuitsMood`) a
ogni mood con quota >= SOGLIA, piu' sempre il suo mood dominante (`ar:dominantMood`).

Serve alla modalita' QUALITATIVA del recommender: dato il mood dell'utente (dallo stadio NLP),
l'ontologia restituisce i generi candidati fra cui scegliere le canzoni.

Stesso principio di `build_genre_taxonomy.py` di AlgoRun (affinita' osservata nel dataset,
non a mano). Fonti teoriche in docs/THEORY.md (Russell 1980; Karageorghis & Terry 2009).

Uso:  python ontology/build_mood_ontology.py [percorso_csv]
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import csv

MOODS = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]
THRESHOLD = 0.25   # quota minima di supporto perche' un genere "serva" un mood (design -> ablation)

BASE = Path(__file__).parent
DEFAULT_CSV = BASE.parent / "songs.csv"
OUT_TTL = BASE / "genre_mood.ttl"


def _iri(label: str) -> str:
    return "ar:g_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def compute(csv_path: Path) -> dict[str, dict]:
    """genere -> {dominant, moods:[...]} dai supports_mood osservati."""
    g_mood: dict[str, Counter] = defaultdict(Counter)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            genre = (row.get("genre") or "").strip()
            if not genre:
                continue
            for tok in (row.get("supports_mood") or "").split(";"):
                tok = tok.strip()
                if tok:
                    g_mood[genre][tok] += 1

    out: dict[str, dict] = {}
    for genre, c in g_mood.items():
        total = sum(c.values())
        shares = {m: c.get(m, 0) / total for m in MOODS}
        dominant = max(shares, key=shares.get)
        suits = [m for m in MOODS if m == dominant or shares[m] >= THRESHOLD]
        out[genre] = {"dominant": dominant, "moods": suits,
                      "shares": {m: round(shares[m], 3) for m in MOODS}}
    return out


def render_ttl(assign: dict[str, dict]) -> str:
    lines = [
        "# GENERATO da ontology/build_mood_ontology.py — non editare a mano.",
        "# Associazione mood -> generi ancorata ai supports_mood osservati in songs.csv.",
        "@prefix ar:   <http://runmaxxin.org/ontology#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "",
        'ar:MoodScheme a skos:ConceptScheme ; rdfs:label "workout mood scheme" .',
        'ar:GenreScheme a skos:ConceptScheme ; rdfs:label "music genre scheme" .',
        "",
        "# --- Mood (vocabolario supports_mood) ---",
    ]
    for m in MOODS:
        lines.append(f'ar:{m} a ar:Mood, skos:Concept ; rdfs:label "{m}" ; skos:inScheme ar:MoodScheme .')
    lines.append("")
    lines.append("# --- Generi (foglia) con affinita' di mood osservata ---")
    for genre in sorted(assign):
        info = assign[genre]
        suits = " , ".join(f"ar:{m}" for m in info["moods"])
        lines.append(
            f'{_iri(genre)} a ar:Genre, skos:Concept ; rdfs:label "{genre}" ; '
            f'skos:inScheme ar:GenreScheme ; '
            f'ar:dominantMood ar:{info["dominant"]} ; '
            f'ar:genreSuitsMood {suits} .'
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    assign = compute(csv_path)
    OUT_TTL.write_text(render_ttl(assign), encoding="utf-8")

    per_mood = Counter()
    for info in assign.values():
        for m in info["moods"]:
            per_mood[m] += 1
    print(f"Ontologia scritta in: {OUT_TTL}")
    print(f"Generi: {len(assign)}  (soglia {THRESHOLD})")
    print("Generi per mood:")
    for m in MOODS:
        print(f"  {m:12s} {per_mood[m]}")


if __name__ == "__main__":
    main()

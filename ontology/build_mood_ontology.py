"""Genera l'ontologia OWL mood -> generi musicali (ontology/genre_mood.owl).

Fornisce uno STRATO SIMBOLICO riusabile (classi Mood/Genre, proprieta', individui) per la
modalita' QUALITATIVA del recommender: dato il mood dell'utente (dallo stadio NLP), l'ontologia
restituisce i generi candidati fra cui scegliere le canzoni.

Le affinita' mood-genere sono POPOLATE EMPIRICAMENTE (ontology population) dalle distribuzioni
di feature osservate nel catalogo: per ogni genere un mood e' associato (`ar:genreSuitsMood`)
se la sua evidenza supera una soglia, piu' sempre il mood dominante (`ar:dominantMood`).

OWL in sintassi Turtle (owl:Class / owl:ObjectProperty / owl:NamedIndividual), stesso stile di
`ontology/algorun.owl`. Fondamenti teorici in docs/THEORY.md
(Russell 1980, valenza x arousal; Karageorghis & Terry 2009; Rada 1989).

Uso:  python ontology/build_mood_ontology.py [percorso_csv]
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MOODS = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]
THRESHOLD = 0.25   # soglia di evidenza minima perche' un genere "serva" un mood (design -> ablation)

BASE = Path(__file__).parent
DEFAULT_CSV = BASE.parent / "songs.csv"
OUT_OWL = BASE / "genre_mood.owl"


def _iri(label: str) -> str:
    return "ar:g_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def compute(csv_path: Path) -> dict[str, dict]:
    """genere -> {dominant, moods:[...], shares:{...}} dall'evidenza empirica osservata."""
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


def render_owl(assign: dict[str, dict]) -> str:
    L = [
        "# Strato simbolico generato da ontology/build_mood_ontology.py — non editare a mano.",
        "# Ontologia OWL mood -> generi; affinita' popolate empiricamente (ontology population).",
        "@prefix ar:   <http://runmaxxin.org/ontology#> .",
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "<http://runmaxxin.org/ontology> a owl:Ontology ;",
        '    rdfs:label "RUNMAXXIN Mood-Genre Ontology" ;',
        '    owl:versionInfo "1.0.0" ;',
        '    rdfs:comment "Ontologia mood -> genere: strato simbolico per la modalita\' qualitativa '
        'del recommender. Affinita\' genreSuitsMood popolate empiricamente (soglia sull\'evidenza '
        "+ mood dominante). Fonti: Russell 1980; Karageorghis & Terry 2009; Rada 1989.\" .",
        "",
        "#################################################################",
        "# Classi",
        "#################################################################",
        'ar:Mood a owl:Class ; rdfs:label "mood" .',
        'ar:Genre a owl:Class ; rdfs:label "music genre" .',
        "ar:Mood owl:disjointWith ar:Genre .",
        "",
        "#################################################################",
        "# Proprieta'",
        "#################################################################",
        "ar:genreSuitsMood a owl:ObjectProperty ;",
        '    rdfs:label "genre suits mood" ; rdfs:domain ar:Genre ; rdfs:range ar:Mood .',
        "ar:dominantMood a owl:ObjectProperty, owl:FunctionalProperty ;",
        '    rdfs:label "dominant mood" ; rdfs:subPropertyOf ar:genreSuitsMood ;',
        "    rdfs:domain ar:Genre ; rdfs:range ar:Mood .",
        "",
        "#################################################################",
        "# Individui: Mood",
        "#################################################################",
    ]
    for m in MOODS:
        L.append(f'ar:{m} a owl:NamedIndividual, ar:Mood ; rdfs:label "{m}" .')
    L += [
        "",
        "#################################################################",
        "# Individui: Genere (affinita' di mood osservata nel dataset)",
        "#################################################################",
    ]
    for genre in sorted(assign):
        info = assign[genre]
        suits = " , ".join(f"ar:{m}" for m in info["moods"])
        L.append(
            f'{_iri(genre)} a owl:NamedIndividual, ar:Genre ; rdfs:label "{genre}" ; '
            f'ar:dominantMood ar:{info["dominant"]} ; '
            f'ar:genreSuitsMood {suits} .'
        )
    return "\n".join(L) + "\n"


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    assign = compute(csv_path)
    OUT_OWL.write_text(render_owl(assign), encoding="utf-8")

    per_mood = Counter()
    for info in assign.values():
        for m in info["moods"]:
            per_mood[m] += 1
    print(f"Ontologia OWL scritta in: {OUT_OWL}")
    print(f"Generi: {len(assign)}  (soglia {THRESHOLD})")
    print("Generi per mood:")
    for m in MOODS:
        print(f"  {m:12s} {per_mood[m]}")


if __name__ == "__main__":
    main()

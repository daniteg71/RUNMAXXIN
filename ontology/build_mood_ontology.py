"""Genera l'ontologia OWL mood -> generi musicali (ontology/genre_mood.owl).

KNOWLEDGE-DRIVEN (non data-driven): le associazioni mood<->genere derivano dalla TEORIA, non
dalle statistiche del catalogo.
  - Modello di Russell (1980): valenza x arousal. Ogni mood e' una regione di quel piano.
  - Karageorghis & Terry (2009): l'arousal della musica dipende da tempo/energia -> ogni
    famiglia di generi ha un archetipo (arousal, valenza) noto dalla letteratura.
  - Regola: un genere "serve" i mood la cui regione di Russell contiene il suo archetipo.

Il CSV `songs.csv` viene letto SOLO per sapere quali generi esistono (istanze A-Box); la
colonna `supports_mood` NON viene usata. La T-Box (classi/proprieta') e le regole di affinita'
sono definite qui a priori. Il CSV puo' poi essere usato per VALIDARE (accordo osservato), non
per costruire l'ontologia.

Uso:  python ontology/build_mood_ontology.py [percorso_csv]
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

MOODS = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]

BASE = Path(__file__).parent
DEFAULT_CSV = BASE.parent / "songs.csv"
OUT_OWL = BASE / "genre_mood.owl"

# --- REGOLA teorica: archetipo (famiglia di generi) -> mood via regioni di Russell ---
# arousal/valenza dell'archetipo (Karageorghis), poi mappati ai mood (Russell).
# ordine = priorita' di match (dal piu' specifico); il 1o archetipo che combacia vince.
ARCHETYPES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    # (nome, parole-chiave nel nome del genere, mood associati)
    ("low_arousal_calm",                                       # bassa attivazione, valenza serena
     ("ambient", "classical", "piano", "sleep", "new-age", "chill", "acoustic", "jazz",
      "opera", "romance", "singer-songwriter", "sad", "blues", "folk", "bluegrass",
      "study", "world", "guitar"),
     ("Calm", "Focused")),
    ("high_arousal_intense",                                   # alta attivazione, valenza bassa/tesa
     ("metal", "hardcore", "grindcore", "hard-rock", "industrial", "goth", "punk",
      "grunge", "emo", "screamo", "thrash"),
     ("Energetic",)),
    ("high_arousal_dance",                                     # alta attivazione, elettronica ballabile
     ("edm", "techno", "house", "trance", "hardstyle", "dubstep", "drum-and-bass",
      "breakbeat", "electro", "electronic", "idm", "dub", "club", "dancehall", "garage"),
     ("Energetic", "Motivated")),
    ("upbeat_positive",                                        # attivazione medio-alta, valenza alta
     ("pop", "disco", "funk", "groove", "salsa", "samba", "reggaeton", "afrobeat",
      "latin", "forro", "pagode", "ska", "party", "happy", "sertanejo", "synth",
      "j-idol", "j-dance", "mpb"),
     ("Motivated", "Energetic")),
    ("mid_driving",                                            # attivazione media, ritmo trainante
     ("rock", "indie", "alternative", "british", "hip-hop", "r-n-b", "trip-hop", "soul"),
     ("Focused", "Motivated")),
]
DEFAULT_MOODS = ("Neutral", "Focused")   # generi non archetipici (lingue, comedy, kids, ...)


def _iri(label: str) -> str:
    return "ar:g_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def classify_genre(genre: str) -> tuple[str, ...]:
    """Regola teorica genere -> mood (via archetipo di Russell/Karageorghis)."""
    low = genre.lower()
    for _name, keywords, moods in ARCHETYPES:
        if any(k in low for k in keywords):
            return moods
    return DEFAULT_MOODS


def catalog_genres(csv_path: Path) -> list[str]:
    """Legge dal CSV SOLO l'insieme dei generi esistenti (istanze A-Box)."""
    genres: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("genre") or "").strip()
            if g:
                genres.add(g)
    return sorted(genres)


def render_owl(assign: dict[str, dict]) -> str:
    L = [
        "# Strato simbolico generato da ontology/build_mood_ontology.py — non editare a mano.",
        "# Ontologia OWL mood -> generi, KNOWLEDGE-DRIVEN (Russell 1980; Karageorghis & Terry 2009).",
        "# Le associazioni derivano dalla teoria, non dalle statistiche del catalogo.",
        "@prefix ar:   <http://runmaxxin.org/ontology#> .",
        "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
        "@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .",
        "",
        "<http://runmaxxin.org/ontology> a owl:Ontology ;",
        '    rdfs:label "RUNMAXXIN Mood-Genre Ontology" ;',
        '    owl:versionInfo "2.0.0" ;',
        '    rdfs:comment "Mood -> genere per la modalita\' qualitativa. genreSuitsMood definito a '
        "priori dalla teoria (Russell: valenza x arousal; Karageorghis: arousal per famiglia di generi). "
        "Il catalogo popola solo le istanze (A-Box), non le regole (T-Box).\" .",
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
        "# Individui: Mood (regioni del piano di Russell)",
        "#################################################################",
    ]
    for m in MOODS:
        L.append(f'ar:{m} a owl:NamedIndividual, ar:Mood ; rdfs:label "{m}" .')
    L += [
        "",
        "#################################################################",
        "# Individui: Genere (istanze dal catalogo; affinita' dalla regola teorica)",
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
    assign = {}
    for genre in catalog_genres(csv_path):
        moods = classify_genre(genre)
        assign[genre] = {"dominant": moods[0], "moods": list(moods)}
    OUT_OWL.write_text(render_owl(assign), encoding="utf-8")

    per_mood = Counter()
    for info in assign.values():
        for m in info["moods"]:
            per_mood[m] += 1
    print(f"Ontologia OWL (knowledge-driven) scritta in: {OUT_OWL}")
    print(f"Generi (istanze A-Box): {len(assign)}")
    print("Generi per mood:")
    for m in MOODS:
        print(f"  {m:12s} {per_mood[m]}")


if __name__ == "__main__":
    main()

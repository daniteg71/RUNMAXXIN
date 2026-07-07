"""Accessor dell'ontologia OWL mood -> generi (ontology/genre_mood.owl) per la modalita' QUALITATIVA.

Dato il mood dell'utente (dallo stadio NLP, `intent.predict_mood`), restituisce i generi
candidati fra cui il recommender sceglie le canzoni quando NON c'e' un BPM target 'chirurgico'.

Parsing pure-Python del nostro OWL/Turtle (formato stabile, una riga per individuo genere) ->
nessuna dipendenza a runtime. Il file resta OWL standard, caricabile con rdflib/Protege.
"""
from __future__ import annotations

import re
from pathlib import Path

_OWL = Path(__file__).parent / "ontology" / "genre_mood.owl"

# es riga: ar:g_happy a ... rdfs:label "happy" ; ... ar:dominantMood ar:Energetic ;
#          ar:genreSuitsMood ar:Energetic , ar:Neutral .
_LABEL = re.compile(r'rdfs:label\s+"([^"]+)"')
_DOMINANT = re.compile(r"ar:dominantMood\s+ar:(\w+)")
_SUITS = re.compile(r"ar:genreSuitsMood\s+([^.]+)\.")

_GENRE_TO_MOODS: dict[str, list[str]] | None = None
_GENRE_DOMINANT: dict[str, str] = {}


def _load() -> dict[str, list[str]]:
    global _GENRE_TO_MOODS
    if _GENRE_TO_MOODS is not None:
        return _GENRE_TO_MOODS
    mapping: dict[str, list[str]] = {}
    for line in _OWL.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ar:g_"):
            continue
        lab = _LABEL.search(line)
        suits = _SUITS.search(line)
        if not lab or not suits:
            continue
        genre = lab.group(1)
        moods = re.findall(r"ar:(\w+)", suits.group(1))
        mapping[genre] = moods
        if (d := _DOMINANT.search(line)):
            _GENRE_DOMINANT[genre] = d.group(1)
    _GENRE_TO_MOODS = mapping
    return mapping


def genres_for_mood(mood: str) -> list[str]:
    """Tutti i generi che 'servono' il mood dato (quota osservata >= soglia o dominante)."""
    return sorted(g for g, moods in _load().items() if mood in moods)


def dominant_genres_for_mood(mood: str) -> list[str]:
    """Solo i generi il cui mood DOMINANTE e' quello dato (segnale piu' forte)."""
    _load()
    return sorted(g for g, dom in _GENRE_DOMINANT.items() if dom == mood)


def moods_for_genre(genre: str) -> list[str]:
    """I mood a cui un genere e' associato."""
    return list(_load().get(genre, []))


if __name__ == "__main__":
    import sys
    mood = sys.argv[1] if len(sys.argv) > 1 else "Energetic"
    gs = genres_for_mood(mood)
    print(f"{mood}: {len(gs)} generi")
    print("  tutti     :", ", ".join(gs))
    print("  dominanti :", ", ".join(dominant_genres_for_mood(mood)))

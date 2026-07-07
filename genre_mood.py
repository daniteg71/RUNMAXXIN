"""Mood -> generi musicali: DIZIONARIO Python, non un'ontologia.

Prima era un file OWL letto a regex — funzionalmente era già un dizionario (nessuna
query, nessuna inferenza, nessun reasoner), solo scritto in un formato più complicato.
Qui è dichiarato per quello che è. La regola resta quella teorica (Russell 1980,
valenza x arousal; Karageorghis & Terry 2009, arousal per famiglia di generi): non è
dedotta dalle statistiche del catalogo. `songs.csv` fornisce solo l'elenco dei generi
esistenti; le associazioni mood<->genere vengono dagli archetipi qui sotto.

Dato il mood dell'utente (dallo stadio NLP, `intent.predict_mood`), restituisce i generi
candidati fra cui il recommender sceglie le canzoni quando NON c'è un BPM target 'chirurgico'.
"""
from __future__ import annotations

import csv
from pathlib import Path

MOODS = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]
SONGS_CSV = Path(__file__).parent / "songs.csv"

# archetipo -> (parole-chiave nel nome del genere, mood associati). Il PRIMO archetipo
# che combacia vince. Fonte: Russell 1980 (piano valenza x arousal) + Karageorghis &
# Terry 2009 (arousal per famiglia musicale) -- regola teorica, non statistica.
ARCHETYPES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
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


def _classify_genre(genre: str) -> tuple[str, ...]:
    """Regola teorica genere -> mood (via archetipo di Russell/Karageorghis)."""
    low = genre.lower()
    for _name, keywords, moods in ARCHETYPES:
        if any(k in low for k in keywords):
            return moods
    return DEFAULT_MOODS


def _catalog_genres() -> list[str]:
    """Legge dal CSV SOLO l'elenco dei generi esistenti (non le associazioni mood)."""
    genres: set[str] = set()
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("genre") or "").strip()
            if g:
                genres.add(g)
    return sorted(genres)


def _build() -> dict[str, dict]:
    return {g: {"dominant": _classify_genre(g)[0], "moods": list(_classify_genre(g))}
            for g in _catalog_genres()}


GENRE_TO_MOODS: dict[str, dict] = _build()   # calcolato una volta all'import


def genres_for_mood(mood: str) -> list[str]:
    """Tutti i generi associati al mood dato (dalla regola teorica)."""
    return sorted(g for g, info in GENRE_TO_MOODS.items() if mood in info["moods"])


def dominant_genres_for_mood(mood: str) -> list[str]:
    """Solo i generi il cui mood DOMINANTE è quello dato (segnale più forte)."""
    return sorted(g for g, info in GENRE_TO_MOODS.items() if info["dominant"] == mood)


def moods_for_genre(genre: str) -> list[str]:
    """I mood a cui un genere è associato."""
    return list(GENRE_TO_MOODS.get(genre, {}).get("moods", []))


if __name__ == "__main__":
    import sys
    mood = sys.argv[1] if len(sys.argv) > 1 else "Energetic"
    gs = genres_for_mood(mood)
    print(f"{mood}: {len(gs)} generi")
    print("  tutti     :", ", ".join(gs))
    print("  dominanti :", ", ".join(dominant_genres_for_mood(mood)))

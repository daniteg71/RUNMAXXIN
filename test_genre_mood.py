"""Test dell'ontologia mood -> generi (deterministici, senza modelli).

Verifica che l'ontologia sia coerente col vocabolario e col catalogo songs.csv:
mood validi, ogni mood non vuoto, generi ⊆ generi del CSV, dominante ∈ associati.
"""
import csv
from pathlib import Path

from intent import MOOD_LABELS
from genre_mood import (dominant_genres_for_mood, genres_for_mood,
                        moods_for_genre)

BASE = Path(__file__).parent
CSV = BASE / "songs.csv"
TTL = BASE / "ontology" / "genre_mood.ttl"


def _csv_genres() -> set[str]:
    with open(CSV, newline="", encoding="utf-8") as f:
        return {(r.get("genre") or "").strip() for r in csv.DictReader(f) if r.get("genre")}


def test_ttl_exists():
    assert TTL.exists(), "ontology/genre_mood.ttl mancante: esegui build_mood_ontology.py"


def test_every_mood_has_genres():
    for mood in MOOD_LABELS:
        assert genres_for_mood(mood), f"nessun genere per mood {mood}"


def test_genres_are_subset_of_csv():
    csv_genres = _csv_genres()
    for mood in MOOD_LABELS:
        for g in genres_for_mood(mood):
            assert g in csv_genres, f"{g} non e' nel catalogo"


def test_dominant_is_subset_of_suited():
    for mood in MOOD_LABELS:
        dom = set(dominant_genres_for_mood(mood))
        suited = set(genres_for_mood(mood))
        assert dom <= suited, f"dominanti di {mood} non ⊆ associati"


def test_moods_for_genre_use_valid_labels():
    for mood in MOOD_LABELS:
        for g in genres_for_mood(mood):
            assert set(moods_for_genre(g)) <= set(MOOD_LABELS)

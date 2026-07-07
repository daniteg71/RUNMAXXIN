#  given the user's mood from the NLP module intent.py (intent.predict_mood),
#  the following code returns the music genres from which songs can be selected when no target BPM is available


#  the overall code works as follows:

#  mood predicted by the NLP module
#          ↓
#  compatible music genres
#          ↓
#  genre list passed to the controller/recommender


#  the mood → music-genre association criterion follows the theoretical rule
#  derived from the studies by Karageorghis & Terry (2009)




#  the following libraries are used:
#    - annotations to use more flexible type annotations
#    - csv to read the songs.csv dataset
#    - Path to build file and folder paths

from __future__ import annotations
import csv
from pathlib import Path



# available moods
MOODS = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]

# song-dataset path
SONGS_CSV = Path(__file__).parent / "songs.csv"


#  the central part of the file is based on archetypes, i.e. music families characterized by
#  a certain level of arousal (degree of activation) and valence (emotional positivity/negativity)
#  each archetype contains three elements (name/genre keywords/associated moods)

ARCHETYPES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("low_arousal_calm",                                       # low arousal, calm valence
     ("ambient", "classical", "piano", "sleep", "new-age", "chill", "acoustic", "jazz",
      "opera", "romance", "singer-songwriter", "sad", "blues", "folk", "bluegrass",
      "study", "world", "guitar"),
     ("Calm", "Focused")),
    ("high_arousal_intense",                                   # high arousal, low/tense valence
     ("metal", "hardcore", "grindcore", "hard-rock", "industrial", "goth", "punk",
      "grunge", "emo", "screamo", "thrash"),
     ("Energetic",)),
    ("high_arousal_dance",                                     # high arousal, dance-oriented electronic music
     ("edm", "techno", "house", "trance", "hardstyle", "dubstep", "drum-and-bass",
      "breakbeat", "electro", "electronic", "idm", "dub", "club", "dancehall", "garage"),
     ("Energetic", "Motivated")),
    ("upbeat_positive",                                        # medium-high arousal, high valence
     ("pop", "disco", "funk", "groove", "salsa", "samba", "reggaeton", "afrobeat",
      "latin", "forro", "pagode", "ska", "party", "happy", "sertanejo", "synth",
      "j-idol", "j-dance", "mpb"),
     ("Motivated", "Energetic")),
    ("mid_driving",                                            # medium arousal, driving rhythm
     ("rock", "indie", "alternative", "british", "hip-hop", "r-n-b", "trip-hop", "soul"),
     ("Focused", "Motivated")),
]
DEFAULT_MOODS = ("Neutral", "Focused")   # non-archetypal genres (languages, comedy, kids, ...)



#  the following function receives a genre name and returns the associated moods
#  according to the theoretical Russell/Karageorghis rule

def _classify_genre(genre: str) -> tuple[str, ...]:
    low = genre.lower()
    for _name, keywords, moods in ARCHETYPES:
        if any(k in low for k in keywords):
            return moods
    return DEFAULT_MOODS


#  the following function reads the genres actually present in
#  the song dataset being used (songs.csv)

def _catalog_genres() -> list[str]:
    genres: set[str] = set()
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("genre") or "").strip()
            if g:
                genres.add(g)
    return sorted(genres)


#  the following function builds the complete genre → associated-moods mapping
#  and stores the genre, its dominant mood, and the list of all associated moods

def _build() -> dict[str, dict]:
    return {g: {"dominant": _classify_genre(g)[0], "moods": list(_classify_genre(g))}
            for g in _catalog_genres()}


# executed when the file is imported
GENRE_TO_MOODS: dict[str, dict] = _build()   


#  the following function returns the list of genres associated with the requested mood
def genres_for_mood(mood: str) -> list[str]:
    return sorted(g for g, info in GENRE_TO_MOODS.items() if mood in info["moods"])


#  the following function returns the genres for which the input mood is dominant
def dominant_genres_for_mood(mood: str) -> list[str]:
    return sorted(g for g, info in GENRE_TO_MOODS.items() if info["dominant"] == mood)


#  the following function returns the list of moods associated with the genre
def moods_for_genre(genre: str) -> list[str]:
    return list(GENRE_TO_MOODS.get(genre, {}).get("moods", []))




if __name__ == "__main__":
    import sys
    mood = sys.argv[1] if len(sys.argv) > 1 else "Energetic"
    gs = genres_for_mood(mood)
    print(f"{mood}: {len(gs)} generi")
    print("  tutti     :", ", ".join(gs))
    print("  dominanti :", ", ".join(dominant_genres_for_mood(mood)))

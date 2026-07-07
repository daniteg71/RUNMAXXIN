#  il seguente codice, dato il mood dell'utente dall' NLP intent.py (intent.predict_mood),
#  restituisce i generi musicali fra cui scegliere le canzoni (quando non c'è un BPM target)


#  il codice nel complesso lavora in questo modo:

#  mood previsto dal modulo NLP
#          ↓
#  generi musicali compatibili
#          ↓
#  lista di generi passata al controller/recommender


#  il criterio di associazione mood → genere musicale segue la regola 
#  teorica ricavata dagli studi di Karageorghis & Terry nel 2009




#  vengono utilizzate le librerie:
#    - annotations per usare annotazioni di tipo più flessibili
#    - csv per leggere il dataset songs.csv
#    - path per costruire i percorsi delle cartelle contenenti i modelli

from __future__ import annotations
import csv
from pathlib import Path



#lista mood disponibili
MOODS = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]

#path dataset canzoni
SONGS_CSV = Path(__file__).parent / "songs.csv"


#  la parte centrale del file verte sugli archetipi, ovvero famiglie musicali caratterizzate da 
#  un certo livello di arousal (gradi di attivazione) e valence(positività/negatività emotiva)
#  ogni archetipo contiene 3 elementi (nome/parole chiave dei generi/mood associati)

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



#  la seguente funzione riceve il nome di un genere in input e restituisce mood associati
#  secondo la regola teorica di Russell/Karageorghis

def _classify_genre(genre: str) -> tuple[str, ...]:
    low = genre.lower()
    for _name, keywords, moods in ARCHETYPES:
        if any(k in low for k in keywords):
            return moods
    return DEFAULT_MOODS


#  la seguente funzione legge i generi effettivamente presenti nel
#  dataset delle canzoni utilizzato (songs.csv)

def _catalog_genres() -> list[str]:
    genres: set[str] = set()
    with open(SONGS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("genre") or "").strip()
            if g:
                genres.add(g)
    return sorted(genres)


#  la seguente funzione costruisce il mapping completo genere → mood associati
#  restituisce genere, mood dominante legato a quel genere, lista di tutti i mood associati

def _build() -> dict[str, dict]:
    return {g: {"dominant": _classify_genre(g)[0], "moods": list(_classify_genre(g))}
            for g in _catalog_genres()}


#riga eseguita quando il file viene importato
GENRE_TO_MOODS: dict[str, dict] = _build()   


#  la seguente funzione restituisce la lista di generi associati al mood richiesto
def genres_for_mood(mood: str) -> list[str]:
    return sorted(g for g, info in GENRE_TO_MOODS.items() if mood in info["moods"])


#  la seguente funzione restituisce i generi nei quali il mood in input è dominante
def dominant_genres_for_mood(mood: str) -> list[str]:
    return sorted(g for g, info in GENRE_TO_MOODS.items() if info["dominant"] == mood)


#  la seguente funzione restituisce la lista di mood a cui il genere è associato
def moods_for_genre(genre: str) -> list[str]:
    return list(GENRE_TO_MOODS.get(genre, {}).get("moods", []))




if __name__ == "__main__":
    import sys
    mood = sys.argv[1] if len(sys.argv) > 1 else "Energetic"
    gs = genres_for_mood(mood)
    print(f"{mood}: {len(gs)} generi")
    print("  tutti     :", ", ".join(gs))
    print("  dominanti :", ", ".join(dominant_genres_for_mood(mood)))

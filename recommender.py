#  il seguente codice riceve il Target prodotto dal file controller.py lo confronta
#  con tutte le canzoni del dataset songs.csv e restituisce le top k-canzoni più adatte


#  il codice nel complesso lavora in questo modo:

#  Target del controller
#          ↓
#  catalogo songs.csv
#          ↓
#  eventuale esclusione canzoni recenti
#          ↓
#  eventuale filtro di genere
#          ↓
#  distanza pesata
#          ↓
#  softmax
#          ↓
#  Top k canzoni



#  vengono utilizzate le librerie:
#    - dataclasses per poter creare ogetti dataclass (classe per contenere dati) e trasformarli in dizionario
#    - typing per indicare che una funzione accetta una sequenza ordinata di valori
#    - numpy per convertire BPM in array e calcolare media, s.d., minimo e massimo, regressione lineare


from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


#  path del dataset songs.csv
ROOT = Path(__file__).resolve().parent
DEFAULT_SONGS_PATH = ROOT / "data" / "songs.csv"

#  colonne necessarie affinchè il recommender funzioni
REQUIRED_COLUMNS = {
    "song_id",
    "title",
    "artist",
    "genre",
    "bpm",
    "energy",
    "valence",
}



#  la seguente funzione restituisce le features del target, ricevendo in input 
#  un oggetto Target nel quale chiama il metodo to_dict o direttamente un dizionario python

def _target_dict(target: Any) -> dict:

    if isinstance(target, Mapping):
        return dict(target)

    if hasattr(target, "to_dict"):
        return target.to_dict()

    return {
        "bpm": target.bpm,
        "energy": target.energy,
        "valence": target.valence,
        "weights": target.weights,
        "bpm_tolerance": target.bpm_tolerance,
        "genres": target.genres,
        "tau": target.tau,
    }


#  la seguente funzione normalizza i generi musicali scrivendoli nello stesso formato

def _normalize_genre(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


#  la seguente funzione carica e pulisce il dataset songs.csv
def load_songs(
    songs_path: str | Path = DEFAULT_SONGS_PATH,
) -> pd.DataFrame:
 
    path = Path(songs_path)

    if not path.exists():
        alternative = ROOT / "songs.csv"
        if alternative.exists():
            path = alternative
        else:
            raise FileNotFoundError(
                f"Catalogo non trovato: {path}")

    songs = pd.read_csv(path)
 
    missing = REQUIRED_COLUMNS - set(songs.columns)
    if missing:
        raise ValueError(
            f"Colonne mancanti in songs.csv: {sorted(missing)}")

    songs = songs.copy()

    #converte parametri in numeri, se non convertibile in NaN
    for column in ("bpm", "energy", "valence"):
        songs[column] = pd.to_numeric(
            songs[column],
            errors="coerce",
        )

    #rimuove canzoni che non hanno informazioni essenziali
    songs = songs.dropna(
        subset=[
            "song_id",
            "title",
            "artist",
            "genre",
            "bpm",
            "energy",
            "valence",
        ]
    )

    #tiene solo canzoni con valori sensati
    songs = songs[
        songs["bpm"].gt(0)
        & songs["energy"].between(0, 1)
        & songs["valence"].between(0, 1)
    ]

    return songs.drop_duplicates(
        subset=["song_id"]
    ).reset_index(drop=True)



#  la seguente funzione calcola una distanza euclidea pesata target-canzone
def weighted_distance(
    songs: pd.DataFrame,
    target: Any,
) -> np.ndarray:

    t = _target_dict(target)
    weights = t["weights"]

    weight_sum = sum(
        float(weights[name])
        for name in ("bpm", "energy", "valence")
    )

    if weight_sum <= 0:
        raise ValueError(
            "La somma dei pesi deve essere positiva."
        )

    #vengono normalizzati i pesi (così anche se non sommano uno tornano ad una scala coerenti)
    w_bpm = float(weights["bpm"]) / weight_sum
    w_energy = float(weights["energy"]) / weight_sum
    w_valence = float(weights["valence"]) / weight_sum

    #viene utilizzata la tolleranza ricavata dal target
    tolerance = float(t["bpm_tolerance"])
    if tolerance <= 0:
        raise ValueError(
            "bpm_tolerance deve essere positivo."
        )

    #i BPM vengono divisi per la tolleranza
    #se piccola la distanza dai BPM conta di più
    delta_bpm = (
        songs["bpm"].to_numpy(dtype=float)
        - float(t["bpm"])
    ) / tolerance

    delta_energy = (
        songs["energy"].to_numpy(dtype=float)
        - float(t["energy"])
    )

    delta_valence = (
        songs["valence"].to_numpy(dtype=float)
        - float(t["valence"])
    )

    #formula distanza Euclidea pesata
    return np.sqrt(
        w_bpm * delta_bpm**2
        + w_energy * delta_energy**2
        + w_valence * delta_valence**2
    )



#  la seguente funzione trasforma le distanze in probabilità
#  (distanza piccola = probabilità alta). il parametro tau indica
#  la temperatura della softmax (se piccola differenze fra distanze pesano molto)

def distance_softmax(
    distances: np.ndarray,
    tau: float,
) -> np.ndarray:

    if tau <= 0:
        raise ValueError("tau deve essere positivo.")

    #questa operazione viene fatta per evitare oveflow/underflow causato da valori molto grandi/piccoli
    logits = -distances / tau
    logits -= logits.max()

    values = np.exp(logits)

    return values / values.sum()


#  funzione principale che ritorna le top-k canzoni ordinate per probabilità

def recommend(
    target: Any,
    songs_path: str | Path = DEFAULT_SONGS_PATH,
    top_k: int = 3,
    exclude_song_ids: Sequence[str] = (),
    candidate_k: int = 20,
    seed: int | None = None,
) -> pd.DataFrame:

    if top_k <= 0:
        raise ValueError("top_k deve essere positivo.")
    
    if candidate_k <= 0:
        raise ValueError("candidate_k deve essere positivo.")

    t = _target_dict(target)
    songs = load_songs(songs_path)

    #esclude le canzoni già ascoltate recentemente
    if exclude_song_ids:
        songs = songs[
            ~songs["song_id"]
            .astype(str)
            .isin({str(x) for x in exclude_song_ids})
        ].copy()

    if songs.empty:
        raise ValueError("Nessuna canzone candidata.")

    #vengono normalizzati generi rendendoli dello stesso formato
    target_genres = {
        _normalize_genre(genre)
        for genre in (t.get("genres") or [])
    }

    #se target contiene generi, indica se la canzone appartiene ad uno di quei generi
    songs["genre_match"] = (
        songs["genre"]
        .map(_normalize_genre)
        .isin(target_genres)
        if target_genres
        else True
    )

    #applica il filtro soltanto se rimangono almeno top_k brani.
    if target_genres:
        compatible = songs[songs["genre_match"]]

        if len(compatible) >= top_k:
            songs = compatible.copy()

    #viene calcolata distanza per ogni canzone candidata
    distances = weighted_distance(
        songs=songs,
        target=t,
    )

    #ogni distanza viene convertita in probabilità
    probabilities = distance_softmax(
        distances=distances,
        tau=float(t["tau"]),
    )

    #vengono aggiunte le seguenti colonne al dataset songs.csv
    songs["distance"] = distances
    songs["probability"] = probabilities
    songs["probability_percent"] = (
        songs["probability"] * 100
    )

    pool_size = min(
        max(top_k, candidate_k),
        len(songs),
    )

    #si crea un pool delle migliori canzoni candidate
    candidate_pool = (
        songs
        .sort_values(
            ["probability", "distance"],
            ascending=[False, True],
        )
        .head(pool_size)
        .copy()
    )

    #si ricavano le probabilità delle canzoni candidate
    pool_probabilities = candidate_pool[
        "probability"
    ].to_numpy(dtype=float)

    probability_sum = pool_probabilities.sum()

    if probability_sum <= 0:
        raise ValueError(
            "Le probabilità del candidate pool non sono normalizzabili."
        )

    #normalizza le probabilità dei candidati sul pool
    pool_probabilities = (
        pool_probabilities
        / probability_sum
    )

    #generatore numeri casuali
    rng = np.random.default_rng(seed)

    #viene deciso quante canzoni estrarre dal pool
    sample_size = min(
        top_k,
        len(candidate_pool),
    )

    #vengono pescati casualmente k indici senza rimpiazzo
    #rispettando la probability distribution del pool
    sampled_positions = rng.choice(
        len(candidate_pool),
        size=sample_size,
        replace=False,
        p=pool_probabilities,
    )

    #vengono prese le canzoni corrispondenti agli indici pescati
    sampled = (
        candidate_pool
        .iloc[sampled_positions]
        .copy()
        .reset_index(drop=True)
    )

    #viene aggiunta colonna sampling_probability all'output
    sampled["sampling_probability"] = (
        pool_probabilities[sampled_positions]
    )


    output_columns = [
        column
        for column in (
            "song_id",
            "title",
            "artist",
            "genre",
            "bpm",
            "energy",
            "valence",
            "danceability",
            "spotify_url",
            "distance",
            "probability",
            "probability_percent",
            "sampling_probability",
        )
        if column in sampled.columns
    ]

    return sampled[output_columns]



# simulazione
if __name__ == "__main__":
    example_target = {
        "bpm": 150.0,
        "energy": 0.75,
        "valence": 0.70,
        "weights": {
            "bpm": 0.80,
            "energy": 0.15,
            "valence": 0.05,
        },
        "bpm_tolerance": 5.0,
        "genres": [],
        "tau": 0.20,
    }

    print(
        recommend(
            target=example_target,
            top_k=3,
            candidate_k=20,
        ).to_string(index=False)
    )

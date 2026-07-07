"""Recommender vettoriale per RUNMAXXIN.

Riceve il Target prodotto da controller.decide(), calcola la distanza pesata
tra target e canzoni e assegna:

    P(song) = exp(-distance / tau) / sum(exp(-distance / tau))

Il catalogo atteso è data/songs.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SONGS_PATH = ROOT / "data" / "songs.csv"

REQUIRED_COLUMNS = {
    "song_id",
    "title",
    "artist",
    "genre",
    "bpm",
    "energy",
    "valence",
}


def _target_dict(target: Any) -> dict:
    """Accetta controller.Target oppure un dizionario."""

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


def _normalize_genre(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "-")
    )


def load_songs(
    songs_path: str | Path = DEFAULT_SONGS_PATH,
) -> pd.DataFrame:
    """Carica e pulisce il catalogo."""

    path = Path(songs_path)

    # Consente anche di tenere songs.csv accanto al file.
    if not path.exists():
        alternative = ROOT / "songs.csv"

        if alternative.exists():
            path = alternative
        else:
            raise FileNotFoundError(
                f"Catalogo non trovato: {path}"
            )

    songs = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(songs.columns)
    if missing:
        raise ValueError(
            f"Colonne mancanti in songs.csv: {sorted(missing)}"
        )

    songs = songs.copy()

    for column in ("bpm", "energy", "valence"):
        songs[column] = pd.to_numeric(
            songs[column],
            errors="coerce",
        )

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

    songs = songs[
        songs["bpm"].gt(0)
        & songs["energy"].between(0, 1)
        & songs["valence"].between(0, 1)
    ]

    return songs.drop_duplicates(
        subset=["song_id"]
    ).reset_index(drop=True)


def weighted_distance(
    songs: pd.DataFrame,
    target: Any,
) -> np.ndarray:
    """Distanza euclidea pesata target-canzone.

    Il BPM viene diviso per bpm_tolerance:
    - tolleranza piccola -> distanza BPM più severa;
    - tolleranza grande -> distanza BPM più permissiva.
    """

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

    w_bpm = float(weights["bpm"]) / weight_sum
    w_energy = float(weights["energy"]) / weight_sum
    w_valence = float(weights["valence"]) / weight_sum

    tolerance = float(t["bpm_tolerance"])
    if tolerance <= 0:
        raise ValueError(
            "bpm_tolerance deve essere positivo."
        )

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

    return np.sqrt(
        w_bpm * delta_bpm**2
        + w_energy * delta_energy**2
        + w_valence * delta_valence**2
    )


def distance_softmax(
    distances: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Calcola P ∝ exp(-distanza/tau) in modo stabile."""

    if tau <= 0:
        raise ValueError("tau deve essere positivo.")

    logits = -distances / tau
    logits -= logits.max()

    values = np.exp(logits)

    return values / values.sum()


def recommend(
    target: Any,
    songs_path: str | Path = DEFAULT_SONGS_PATH,
    top_k: int = 3,
    exclude_song_ids: Sequence[str] = (),
) -> pd.DataFrame:
    """Restituisce le Top-K canzoni ordinate per probabilità."""

    if top_k <= 0:
        raise ValueError("top_k deve essere positivo.")

    t = _target_dict(target)
    songs = load_songs(songs_path)

    # Sliding-window memory: esclude le tracce recenti.
    if exclude_song_ids:
        songs = songs[
            ~songs["song_id"]
            .astype(str)
            .isin({str(x) for x in exclude_song_ids})
        ].copy()

    if songs.empty:
        raise ValueError("Nessuna canzone candidata.")

    # Nel qualitativo controller.py fornisce generi compatibili.
    target_genres = {
        _normalize_genre(genre)
        for genre in (t.get("genres") or [])
    }

    songs["genre_match"] = (
        songs["genre"]
        .map(_normalize_genre)
        .isin(target_genres)
        if target_genres
        else True
    )

    # Applica il filtro soltanto se rimangono almeno top_k brani.
    if target_genres:
        compatible = songs[songs["genre_match"]]

        if len(compatible) >= top_k:
            songs = compatible.copy()

    distances = weighted_distance(
        songs=songs,
        target=t,
    )

    probabilities = distance_softmax(
        distances=distances,
        tau=float(t["tau"]),
    )

    songs["distance"] = distances
    songs["probability"] = probabilities
    songs["probability_percent"] = (
        songs["probability"] * 100
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
        )
        if column in songs.columns
    ]

    return (
        songs
        .sort_values(
            ["probability", "distance"],
            ascending=[False, True],
        )
        .head(top_k)
        [output_columns]
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # Test autonomo. Nell'app reale target arriva da controller.decide().
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
        ).to_string(index=False)
    )

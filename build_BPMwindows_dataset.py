#  il seguente codice prende il CSV con i BPM registrati ogni secondo
#  e lo trasforma in un nuovo dataset in cui ogni riga riassume una finestra
#  temporale di 30 secondi


# il codice nel complesso lavora in questo modo:

#  bpm_sessions.csv
#  una riga per ogni secondo
#          ↓
#  divisione in finestre da 30 secondi
#          ↓
#  calcolo di media, deviazione, HRR, trend, stato
#          ↓
#  physiological_windows.csv
#  una riga per ogni finestra




#  vengono utilizzate le librerie:
#    - argparse permette di passare parametri dal terminale (es. cambiare finestra senza modificare codice)
#    - Path per rappresentare e gestire percorsi file/cartelle
#    - pandas perleggere il CSV, raggruppare sessioni, selezionare finestre e creare nuovo dataset


from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physiological_state import (
    analyze_bpm_window,
)



#  valori tuilizzati per indicare durata di ogni finestra e quanto si sposta ogni volta

DEFAULT_WINDOW_SECONDS = 30
DEFAULT_STRIDE_SECONDS = 5



#  la seguente funzione verifica che il CSV letto abbia la struttura corretta

def validate_input_dataframe(
    df: pd.DataFrame,
) -> None:
    required_columns = {
        "session_id",
        "user_id",
        "second",
        "bpm",
        "resting_hr",
        "max_hr",
        "workout_goal",
        "phase",
    }

    missing_columns = (
        required_columns.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(
            "The input dataset is empty."
        )




#  funzione principale che riceve dati di una singola sessione, dimensione e spostamento finestre
#  ritorna una lista di dizionari dove ogni dizionario rappresenta una finestra


def process_session(
    session_df: pd.DataFrame,
    window_seconds: int,
    stride_seconds: int,
) -> list[dict[str, object]]:
    session_df = (
        session_df
        #ordina finestre in base al secondo
        .sort_values("second")
        #crea nuovi indici per ogni finestra
        .reset_index(drop=True)
    )

    session_id = str(
        session_df["session_id"].iloc[0]
    )

    user_id = str(
        session_df["user_id"].iloc[0]
    )

    resting_values = (
        session_df["resting_hr"]
        .dropna()
        .unique()
    )

    max_values = (
        session_df["max_hr"]
        .dropna()
        .unique()
    )

    if len(resting_values) != 1:
        raise ValueError(
            f"Session {session_id} has multiple resting_hr values."
        )

    if len(max_values) != 1:
        raise ValueError(
            f"Session {session_id} has multiple max_hr values."
        )

    resting_hr = float(resting_values[0])
    max_hr = float(max_values[0])

    rows: list[dict[str, object]] = []

    #vengono create le finestre temporali
    for start_index in range(
        0,
        len(session_df) - window_seconds + 1,
        stride_seconds,
    ):
        end_index = (
            start_index + window_seconds
        )

        window = session_df.iloc[
            start_index:end_index
        ]

        #prende colonna BPM della finestra e la trasforma in array NumPY
        bpm_values = (
            window["bpm"]
            .astype(float)
            .to_numpy()
        )

        #viene chiamata funzione definita in physiological_state.py
        #restituisce oggetto PhysiologicalAnalysis
        analysis = analyze_bpm_window(
            bpm_values=window["bpm"].to_numpy(),
            speed_values=window["speed_kmh"].to_numpy(),
            cadence_values=window["cadence_spm"].to_numpy(),
            resting_hr=resting_hr,
            max_hr=max_hr,
            sampling_rate_hz=1.0,
        )

        #creazione dizionario che descrive la finestra
        row: dict[str, object] = {
            "session_id": session_id,
            "user_id": user_id,
            "window_start_second": int(
                window["second"].iloc[0]
            ),
            "window_end_second": int(
                window["second"].iloc[-1]
            ),
            "resting_hr": resting_hr,
            "max_hr": max_hr,
            "workout_goal": str(
                window["workout_goal"].iloc[-1]
            ),
        }


        #unisce dizionario row e dizionario analysis
        row.update(
            analysis.to_dict()
        )

        rows.append(row)

    #quando tutte le finestre della sessione sono state analizzate, viene restituita la lista
    return rows



#  la seguente funzione coordina il processo, riceve:
#     - percorso file originale
#     - percorso di output
#     - dimensione della finestra
#     - sliding della finestra

#  viene restituito il DataFrame finale


def build_dataset(
    input_path: Path,
    output_path: Path,
    window_seconds: int,
    stride_seconds: int,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )
    
    #caica file in un dataframe
    df = pd.read_csv(input_path)

    #controlla se valido
    validate_input_dataframe(df)

    #crea lista complessiva contenente finestre di tutte le sessioni
    all_rows: list[dict[str, object]] = []

    #viene diviso il dataset in gruppi diversi in base al session_id
    grouped_sessions = df.groupby(
        "session_id",
        sort=True,
    )

    for session_id, session_df in grouped_sessions:

        #costruisce tutte le finestre della sessione
        session_rows = process_session(
            session_df=session_df,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )

        #aggiunge tutti gli elementi di session_rows alla lista generale
        all_rows.extend(session_rows)

        print(
            f"{session_id}: "
            f"{len(session_rows)} windows"
        )

    #trasforma lista dizionari in tabella dove ogni colonna è una chiave e ogni riga un dizionario
    result_df = pd.DataFrame(all_rows)


    #se sessione dura meno della finestra -> errore
    if result_df.empty:
        raise RuntimeError(
            "No windows were generated."
        )


    #crea cartella output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    #salvataggio DataFrame
    result_df.to_csv(
        output_path,
        index=False,
    )

    return result_df




# la seguente funzione viene eseguita quando viene avviato il file da terminale

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create physiological feature windows "
            "from simulated BPM sessions."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/simulated/bpm_sessions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/physiological_windows.csv"
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
    )

    parser.add_argument(
        "--stride-seconds",
        type=int,
        default=DEFAULT_STRIDE_SECONDS,
    )

    args = parser.parse_args()

    if args.window_seconds < 2:
        raise ValueError(
            "window-seconds must be at least 2."
        )

    if args.stride_seconds < 1:
        raise ValueError(
            "stride-seconds must be at least 1."
        )

    result_df = build_dataset(
        input_path=args.input,
        output_path=args.output,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    print(
        "\nDataset saved to: "
        f"{args.output}"
    )

    print(
        f"Shape: {result_df.shape}"
    )

    print("\nEffort-state distribution:")
    print(
        result_df["effort_state"]
        .value_counts()
        .to_string()
    )

    print("\nTrend-state distribution:")
    print(
        result_df["trend_state"]
        .value_counts()
        .to_string()
    )

    print("\nFirst rows:")
    print(
        result_df.head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

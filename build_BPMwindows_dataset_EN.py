#  the following code takes the CSV containing BPM values recorded every second
#  and transforms it into a new dataset in which each row summarizes a
#  30-second time window


# the overall code works as follows:

#  bpm_sessions.csv
#  one row for each second
#          ↓
#  split into 30-second windows
#          ↓
#  calculation of mean, standard deviation, HRR, trend, and state
#          ↓
#  physiological_windows.csv
#  one row for each window




#  the following libraries are used:
#    - argparse allows parameters to be passed from the terminal (e.g. changing the window without modifying the code)
#    - Path represents and manages file/folder paths
#    - pandas reads the CSV, groups sessions, selects windows, and creates the new dataset


from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physiological_state import (
    analyze_bpm_window,
)



#  values used to define the duration of each window and how far it moves each time

DEFAULT_WINDOW_SECONDS = 30
DEFAULT_STRIDE_SECONDS = 5



#  the following function checks that the input CSV has the correct structure

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




#  main function that receives the data of a single session, the window size, and the stride
#  and returns a list of dictionaries, where each dictionary represents one window


def process_session(
    session_df: pd.DataFrame,
    window_seconds: int,
    stride_seconds: int,
) -> list[dict[str, object]]:
    session_df = (
        session_df
        # sorts samples by second
        .sort_values("second")
        # creates new sequential indices
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

    # creates the time windows
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

        # takes the BPM column of the window and converts it into a NumPy array
        bpm_values = (
            window["bpm"]
            .astype(float)
            .to_numpy()
        )

        # calls the function defined in physiological_state.py
        # returns a PhysiologicalAnalysis object
        analysis = analyze_bpm_window(
            bpm_values=window["bpm"].to_numpy(),
            speed_values=window["speed_kmh"].to_numpy(),
            cadence_values=window["cadence_spm"].to_numpy(),
            resting_hr=resting_hr,
            max_hr=max_hr,
            sampling_rate_hz=1.0,
        )

        # creates the dictionary describing the window
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


        # merges the row dictionary with the analysis dictionary
        row.update(
            analysis.to_dict()
        )

        rows.append(row)

    # after all windows in the session have been analyzed, the list is returned
    return rows



#  the following function coordinates the process and receives:
#     - original file path
#     - output path
#     - window size
#     - window stride

#  the final DataFrame is returned


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
    
    # loads the file into a DataFrame
    df = pd.read_csv(input_path)

    # checks whether it is valid
    validate_input_dataframe(df)

    # creates the overall list containing the windows from all sessions
    all_rows: list[dict[str, object]] = []

    # splits the dataset into groups according to session_id
    grouped_sessions = df.groupby(
        "session_id",
        sort=True,
    )

    for session_id, session_df in grouped_sessions:

        # builds all windows for the session
        session_rows = process_session(
            session_df=session_df,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )

        # adds all elements of session_rows to the overall list
        all_rows.extend(session_rows)

        print(
            f"{session_id}: "
            f"{len(session_rows)} windows"
        )

    # converts the list of dictionaries into a table where each key becomes a column and each dictionary becomes a row
    result_df = pd.DataFrame(all_rows)


    # raises an error if no session is long enough to produce a window
    if result_df.empty:
        raise RuntimeError(
            "No windows were generated."
        )


    # creates the output folder
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # saves the DataFrame
    result_df.to_csv(
        output_path,
        index=False,
    )

    return result_df




# the following function is executed when the file is run from the terminal

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
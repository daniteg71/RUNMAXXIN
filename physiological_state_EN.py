#  the following code takes a BPM sequence and transforms it into
#  a physiological summary of the window


#  the overall code works as follows:

#   BPM sequence
#       ↓
#   data validation
#       ↓
#   window statistics
#       ↓
#   trend calculation
#       ↓ 
#   HRR calculation
#       ↓
#   effort classification
#       ↓
#   trend classification
#       ↓
#   final object containing all results



#  the following libraries are used:
#    - dataclasses to create dataclass objects (classes used to store data) and convert them into dictionaries
#    - typing to indicate that a function accepts an ordered sequence of values
#    - NumPy to convert BPM values into arrays and calculate mean, standard deviation, minimum, maximum, and linear regression



from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np



#  thresholds used to convert the HRR value into a category
#  according to the following logic:

#  HRR < 0.40        → LowEffort
#  HRR < 0.70        → TargetEffort
#  HRR < 0.85        → HighEffort
#  HRR >= 0.85       → VeryHighEffort


LOW_EFFORT_LIMIT = 0.40
TARGET_EFFORT_LIMIT = 0.70
HIGH_EFFORT_LIMIT = 0.85

# this threshold is used to determine whether BPM is increasing, decreasing, or stable
# it corresponds to a 3 BPM-per-minute trend threshold
DEFAULT_TREND_THRESHOLD_BPM_PER_SECOND = 0.05



# this class contains the final analysis result

# @dataclass automatically generates the constructor
# frozen=True means that the object cannot be modified after creation
@dataclass(frozen=True)
class PhysiologicalAnalysis:
    # Heart rate
    current_bpm: float
    mean_bpm: float
    std_bpm: float
    min_bpm: float
    max_bpm: float
    delta_bpm: float
    slope_bpm_per_second: float
    slope_bpm_per_minute: float

    # HRR and classifications
    current_hrr: float
    mean_hrr: float
    effort_state: str
    trend_state: str

    # Speed
    current_speed_kmh: float
    mean_speed_kmh: float
    std_speed_kmh: float
    min_speed_kmh: float
    max_speed_kmh: float
    slope_speed_kmh_per_second: float

    # Cadence
    current_cadence_spm: float
    mean_cadence_spm: float
    std_cadence_spm: float
    min_cadence_spm: float
    max_cadence_spm: float
    slope_cadence_spm_per_second: float

    # converts the object into a Python dictionary
    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)




# the following function checks that the heart-rate profile values are valid

def validate_heart_rate_profile(
    resting_hr: float,
    max_hr: float,
) -> None:
    if resting_hr <= 0:
        raise ValueError("resting_hr must be positive.")

    if max_hr <= 0:
        raise ValueError("max_hr must be positive.")

    if max_hr <= resting_hr:
        raise ValueError(
            "max_hr must be greater than resting_hr."
        )


# this function calculates HRR (Heart Rate Reserve)
# HRR = (HR - resting_HR) / (max_HR - resting_HR)

def compute_hrr(
    heart_rate: float,
    resting_hr: float,
    max_hr: float,
) -> float:


    validate_heart_rate_profile(
        resting_hr=resting_hr,
        max_hr=max_hr,
    )

    hrr = (
        heart_rate - resting_hr
    ) / (
        max_hr - resting_hr
    )
    
    # limits the result to [0.0, 1.2] rather than [0, 1] to tolerate measurement errors
    return float(
        np.clip(hrr, 0.0, 1.2)
    )



#  the following function receives an HRR value and returns a category

def classify_effort(
    hrr: float,
) -> str:
    if hrr < LOW_EFFORT_LIMIT:
        return "LowEffort"

    if hrr < TARGET_EFFORT_LIMIT:
        return "TargetEffort"

    if hrr < HIGH_EFFORT_LIMIT:
        return "HighEffort"

    return "VeryHighEffort"


#  the following function calculates how quickly the BPM values are increasing or decreasing

def calculate_linear_slope(
    values: np.ndarray,
    sampling_rate_hz: float,
) -> float:

    if sampling_rate_hz <= 0:
        raise ValueError(
            "sampling_rate_hz must be positive."
        )

    if len(values) < 2:
        return 0.0

    seconds = (
        np.arange(len(values), dtype=float)
        / sampling_rate_hz
    )

    # calculates a linear regression and returns the slope
    slope, _ = np.polyfit(
        seconds,
        values,
        deg=1,
    )

    return float(slope)



#  the following function receives the slope and compares it with a threshold
#  and classifies the trend

def classify_trend(
    slope_bpm_per_second: float,
    threshold: float = (
        DEFAULT_TREND_THRESHOLD_BPM_PER_SECOND
    ),
) -> str:
    if threshold <= 0:
        raise ValueError(
            "The trend threshold must be positive."
        )

    if slope_bpm_per_second > threshold:
        return "Increasing"

    if slope_bpm_per_second < -threshold:
        return "Decreasing"

    return "Stable"


#  the following function converts a sequence of sensor values into a NumPy array
#  and checks that the data are valid

def prepare_sensor_values(
    sensor_values: Sequence[float],
    sensor_name: str,
) -> np.ndarray:

    values = np.asarray(
        sensor_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            f"{sensor_name} must be a one-dimensional sequence."
        )

    if len(values) < 2:
        raise ValueError(
            f"At least two {sensor_name} values are required."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{sensor_name} contains NaN or infinite values."
        )

    if np.any(values < 0):
        raise ValueError(
            f"{sensor_name} cannot contain negative values."
        )

    return values




#  this is the main function; it receives:

#     - BPM sequence
#     - speed values
#     - cadence values
#     - resting heart rate
#     - maximum heart rate
#     - sampling rate

#  and returns a PhysiologicalAnalysis object


def analyze_bpm_window(
    bpm_values: Sequence[float],
    speed_values: Sequence[float],
    cadence_values: Sequence[float],
    resting_hr: float,
    max_hr: float,
    sampling_rate_hz: float = 1.0,
) -> PhysiologicalAnalysis:


    values = np.asarray(
        bpm_values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "bpm_values must be a one-dimensional sequence."
        )

    if len(values) < 2:
        raise ValueError(
            "At least two BPM values are required."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "bpm_values contains NaN or infinite values."
        )

    if np.any(values <= 0):
        raise ValueError(
            "All BPM values must be positive."
        )
    
    speed = prepare_sensor_values(
        sensor_values=speed_values,
        sensor_name="speed_values",
    )

    cadence = prepare_sensor_values(
        sensor_values=cadence_values,
        sensor_name="cadence_values",
    )

    validate_heart_rate_profile(
        resting_hr=resting_hr,
        max_hr=max_hr,
    )

    current_bpm = float(values[-1])
    mean_bpm = float(np.mean(values))
    std_bpm = float(np.std(values))
    min_bpm = float(np.min(values))
    max_bpm = float(np.max(values))
    delta_bpm = float(values[-1] - values[0])
    current_speed_kmh = float(speed[-1])
    mean_speed_kmh = float(np.mean(speed))
    std_speed_kmh = float(np.std(speed))
    min_speed_kmh = float(np.min(speed))
    max_speed_kmh = float(np.max(speed))
    current_cadence_spm = float(cadence[-1])
    mean_cadence_spm = float(np.mean(cadence))
    std_cadence_spm = float(np.std(cadence))
    min_cadence_spm = float(np.min(cadence))
    max_cadence_spm = float(np.max(cadence))

    slope_cadence_spm_per_second = calculate_linear_slope(
        values=cadence,
        sampling_rate_hz=sampling_rate_hz,
    )

    slope_speed_kmh_per_second = calculate_linear_slope(
        values=speed,
        sampling_rate_hz=sampling_rate_hz,
    )

    slope_bpm_per_second = calculate_linear_slope(
        values=values,
        sampling_rate_hz=sampling_rate_hz,
    )

    slope_bpm_per_minute = (
        slope_bpm_per_second * 60.0
    )

    current_hrr = compute_hrr(
        heart_rate=current_bpm,
        resting_hr=resting_hr,
        max_hr=max_hr,
    )

    mean_hrr = compute_hrr(
        heart_rate=mean_bpm,
        resting_hr=resting_hr,
        max_hr=max_hr,
    )

    effort_state = classify_effort(
        hrr=current_hrr,
    )

    trend_state = classify_trend(
        slope_bpm_per_second=slope_bpm_per_second,
    )

    # stores all results in a single object
    return PhysiologicalAnalysis(
        current_bpm=current_bpm,
        mean_bpm=mean_bpm,
        std_bpm=std_bpm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        delta_bpm=delta_bpm,
        slope_bpm_per_second=slope_bpm_per_second,
        slope_bpm_per_minute=slope_bpm_per_minute,
        current_hrr=current_hrr,
        mean_hrr=mean_hrr,
        effort_state=effort_state,
        trend_state=trend_state,

        current_speed_kmh=current_speed_kmh,
        mean_speed_kmh=mean_speed_kmh,
        std_speed_kmh=std_speed_kmh,
        min_speed_kmh=min_speed_kmh,
        max_speed_kmh=max_speed_kmh,
        slope_speed_kmh_per_second=(
            slope_speed_kmh_per_second
        ),

        current_cadence_spm=current_cadence_spm,
        mean_cadence_spm=mean_cadence_spm,
        std_cadence_spm=std_cadence_spm,
        min_cadence_spm=min_cadence_spm,
        max_cadence_spm=max_cadence_spm,
        slope_cadence_spm_per_second=(
            slope_cadence_spm_per_second
        ),
    )



#  demo function used to test the code
def demo() -> None:
    example_bpm = [
        145, 147, 148, 150, 151,
        153, 155, 157, 160, 163,
    ]

    example_speed = [
        9.8, 9.9, 10.0, 10.1, 10.2,
        10.3, 10.4, 10.5, 10.6, 10.7,
    ]

    example_cadence = [
        158, 159, 160, 160, 161,
        162, 163, 164, 165, 166,
    ]

    result = analyze_bpm_window(
        bpm_values=example_bpm,
        speed_values=example_speed,
        cadence_values=example_cadence,
        resting_hr=60,
        max_hr=195,
        sampling_rate_hz=1.0,
    )

    print("Sensor-window analysis:")

    for key, value in result.to_dict().items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


# runs demo() when the script is executed directly
if __name__ == "__main__":
    demo()
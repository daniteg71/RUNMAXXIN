#  il seguente codice prende una sequenza di BPM e la trasforma in 
#  un riassunto fisiologico della finestra


#  il codice nel complesso lavora in questo modo:

#   sequenza BPM
#       ↓
#   controlli sui dati
#       ↓
#   statistiche della finestra
#       ↓
#   calcolo del trend
#       ↓ 
#   calcolo HRR
#       ↓
#   classificazione dello sforzo
#       ↓
#   classificazione del trend
#       ↓
#   oggetto finale con tutti i risultati



#  vengono utilizzate le librerie:
#    - dataclasses per poter creare ogetti dataclass (classe per contenere dati) e trasformarli in dizionario
#    - typing per indicare che una funzione accetta una sequenza ordinata di valori
#    - numpy per convertire BPM in array e calcolare media, s.d., minimo e massimo, regressione lineare



from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np



#  soglie utilizzate per trasformare il valore HRR in una categoria
#  secondo la seguente logica:

#  HRR < 0.40        → LowEffort
#  HRR < 0.70        → TargetEffort
#  HRR < 0.85        → HighEffort
#  HRR >= 0.85       → VeryHighEffort


LOW_EFFORT_LIMIT = 0.40
TARGET_EFFORT_LIMIT = 0.70
HIGH_EFFORT_LIMIT = 0.85

# questa soglia serve a capire se il BPM sta aumentando/diminuendo/stabile
# equivale a 3 BPM (es. se +3 BPM rispetto a prima, allora "increasing")
DEFAULT_TREND_THRESHOLD_BPM_PER_SECOND = 0.05



# questa classe contiene il risultato finale dell'analisi

# @dataclass genera automaticamente il costruttore
# frozen=True indica che dopo la creazione l'oggetto non può essere modificato
@dataclass(frozen=True)
class PhysiologicalAnalysis:
    # Frequenza cardiaca
    current_bpm: float
    mean_bpm: float
    std_bpm: float
    min_bpm: float
    max_bpm: float
    delta_bpm: float
    slope_bpm_per_second: float
    slope_bpm_per_minute: float

    # HRR e classificazioni
    current_hrr: float
    mean_hrr: float
    effort_state: str
    trend_state: str

    # Velocità
    current_speed_kmh: float
    mean_speed_kmh: float
    std_speed_kmh: float
    min_speed_kmh: float
    max_speed_kmh: float
    slope_speed_kmh_per_second: float

    # Cadenza
    current_cadence_spm: float
    mean_cadence_spm: float
    std_cadence_spm: float
    min_cadence_spm: float
    max_cadence_spm: float
    slope_cadence_spm_per_second: float

    # l'oggetto viene trasformato in un dizionario python
    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)




# la seguente funzione controlla che i valori del profilo siano sensati

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


# questa funzione calcola la HRR (Heart Rate Reserve) utilizzata
# HRR=(HR-HR_rest)/(HR_max-HR_rest)

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
    
    #risultato limitato fra 0.0 e 1.2 (non 1 per tollerare errori)
    return float(
        np.clip(hrr, 0.0, 1.2)
    )



#  la seguente funzione riceve un valore HRR e restituisce una categoria

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


#  la seguente funzione calcola quanto rapidamente il BPM sta aumentando/diminuendo

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

    #calcola regressione lineare e restituisce pendenza (coefficiente angolare)
    slope, _ = np.polyfit(
        seconds,
        values,
        deg=1,
    )

    return float(slope)



#  la seguente funzione riceve la pendenza, la confronta con una soglia
#  e classifica l'andamento 

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


#  la seguente funzione converte una sequenza di valori sensoriali in un array NumPY
#  e controlla che i dati siano validi

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




#  questa è la funzione principale, riceve:

#     - sequenza BPM
#     - valori velocità
#     - valori cadenza
#     - HR a riposo
#     - BPM massimo
#     - frequenza di campionamento

#  e restituisce la PhysiologicalAnalysis


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

    #tutti i risultati inseriti in un unico oggetto
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



#  funzione demo per provare il codice
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


#se viene avviato direttamente lo script esegue demo()
if __name__ == "__main__":
    demo()
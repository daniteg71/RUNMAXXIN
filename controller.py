"""Controller di RUNMAXXIN: genera il VETTORE TARGET per il recommender.

Il recommender (collega) riceve un target e assegna a ogni canzone una probabilita' in base
alla DISTANZA dal target. Questo file produce quel target, fondendo:
  - le feature del prompt iniziale (intent.route: goal, mood, target_bpm, params)
  - i dati che arrivano dai sensori (physiological_state: mean_hrr, effort_state, trend_state)

Regole teoriche:
  - QUANTITATIVO (velocita' dichiarata): target stretto attorno al bpm calcolato (Van Dyck),
    tau basso -> exploit (resta preciso).
  - QUALITATIVO: range piu' largo, generi ristretti al mood (genre_mood), tau piu' alto ->
    explore (variabilita' musicale; softmax di Sutton & Barto).
  - CONSAPEVOLE DEL TIPO: su IntenseRun/ripetute alterna veloce/lento rispetto alla canzone
    precedente (se l'ultima era veloce, punta piu' lento).
  - Il cuore comanda: safety override a HRR alta -> vettore di recupero. La soglia non
    e' una costante Python: e' un dato dichiarato nell'ontologia (ontology/runner_state.owl,
    ar:CriticalState ar:hasThreshold), e la classificazione la fa una query SPARQL
    (symbolic.is_critical_state) -- non un confronto scritto a mano qui.

Funzione pura, nessun loop, nessun recommender: solo decide() -> Target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from intent import GOAL_PARAMS, GOAL_TO_EFFORT, bpm_from_speed
from genre_mood import genres_for_mood
from symbolic import is_critical_state

# --- costanti (design -> ablation, vedi docs/THEORY.md) ---
NARROW = 5.0             # raggio bpm stretto (regime quantitativo): +/- 5 bpm
TAU_EXPLOIT = 0.2        # temperatura bassa = sfrutta (preciso); alta = esplora (varia)
CALM = 0.7               # fattore "calmati" (dici easy ma sali)
PUSH = 1.2               # fattore "spingi"  (dici intense ma vai piano)
RECOVERY_BPM = GOAL_PARAMS["EasyRun"]["bpm"][0]   # in recupero scendi sempre a ~banda facile
WARMUP_MIN = 5.0         # riscaldamento: i primi minuti parti basso e sali fino al target
ENTRAIN_MIN, ENTRAIN_MAX = 150.0, 190.0   # banda naturale di cadenza (Van Dyck): il target la insegue
VAR_DELTA = 8.0          # ampiezza dell'alternanza veloce/lento sulle ripetute (attorno al bpm live)

# mood -> valenza target (Russell 1980; arousal/positivita' della musica)
VALENCE_BY_MOOD = {"Energetic": 0.75, "Motivated": 0.70, "Focused": 0.45,
                   "Neutral": 0.50, "Calm": 0.25}


@dataclass
class Target:
    """Vettore target + metadati che il recommender consuma."""
    bpm: float
    energy: float
    valence: float
    weights: dict            # peso per dimensione nella distanza
    bpm_tolerance: float     # raggio attorno al bpm (stretto/largo)
    genres: list             # generi ammessi (vuoto = nessun filtro)
    tau: float               # temperatura exploration/exploitation del softmax
    mood: str
    goal: str
    effort_band: tuple       # classi matches_effort ammesse
    recovery: bool
    regime: str              # "quantitative" | "qualitative" | "recovery"

    def as_vector(self) -> list:
        """Il punto nello spazio delle canzoni: [bpm, energy, valence]."""
        return [self.bpm, self.energy, self.valence]

    def to_dict(self) -> dict:
        return asdict(self)


def _get(analysis, key, default=None):
    """Legge un campo da PhysiologicalAnalysis (attributo) o da un dict finestra."""
    if analysis is None:
        return default
    if isinstance(analysis, dict):
        return analysis.get(key, default)
    return getattr(analysis, key, default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def live_entrainment_bpm(analysis) -> float | None:
    """BPM che INSEGUE la corsa reale, di momento in momento (entrainment di Van Dyck, continuo).

    UNICO punto di verità per "la musica segue la velocità": preferisce la **cadenza misurata**
    (1:1 coi passi, la cosa che il beat deve inseguire); in mancanza la stima dalla **velocità**
    con `bpm_from_speed` (134 + 2.9·v). Ritorna None se non c'è segnale (cold start) -> il
    chiamante ripiega sul centro-banda del tipo.

    Chiunque voglia "far seguire il bpm alla velocità" chiama QUESTA — così la logica non è
    duplicata: cambiarla qui la cambia ovunque (controller, tester, loop).
    """
    cadence = _get(analysis, "mean_cadence_spm")
    if cadence is not None and float(cadence) > 0:
        return _clamp(float(cadence), ENTRAIN_MIN, ENTRAIN_MAX)     # cadenza misurata, 1:1
    speed = _get(analysis, "mean_speed_kmh")
    if speed is not None and float(speed) > 0:
        return float(bpm_from_speed(float(speed)))                 # stima dalla velocità (Van Dyck)
    return None


def decide(intent: dict, analysis=None, last_bpm: float | None = None,
           elapsed_min: float | None = None) -> Target:
    """Frase (intent) + stato sensori (analysis) -> vettore Target per il recommender.

    `analysis` None = cold start (nessun sensore ancora). `last_bpm` = bpm della canzone
    precedente (variazione veloce/lento su IntenseRun). `elapsed_min` = minuti dall'inizio:
    nei primi WARMUP_MIN il target sale gradualmente da basso al valore previsto (riscaldamento).
    """
    goal = intent.get("goal") or "ModerateRun"
    mood = intent.get("mood") or "Neutral"
    params = intent.get("params") or GOAL_PARAMS[goal]
    lo, hi = params["bpm"]
    target_bpm = intent.get("target_bpm")
    quantitative = target_bpm is not None

    mean_hrr = _get(analysis, "mean_hrr")
    effort = _get(analysis, "effort_state")
    trend = _get(analysis, "trend_state")

    # 1) SAFETY OVERRIDE: il cuore vince su tutto -> recupero.
    # La soglia la dichiara l'ontologia, la classificazione la fa una query SPARQL
    # (symbolic.is_critical_state) -- non un confronto Python scritto a mano.
    if mean_hrr is not None and is_critical_state(mean_hrr):
        return Target(bpm=float(RECOVERY_BPM), energy=min(params["energy"], 0.30), valence=0.25,
                      weights={"bpm": 0.8, "energy": 0.15, "valence": 0.05},
                      bpm_tolerance=NARROW, genres=[], tau=TAU_EXPLOIT,
                      mood=mood, goal=goal, effort_band=("LowEffort",),
                      recovery=True, regime="recovery")

    # 2) regime -> bpm base, raggio, tau, generi, pesi
    if quantitative:
        bpm = float(target_bpm)
        tol = NARROW
        tau = TAU_EXPLOIT
        genres: list = []
        weights = {"bpm": 0.8, "energy": 0.15, "valence": 0.05}
    else:
        # QUALITATIVO: il BPM INSEGUE la velocità reale del momento (entrainment continuo).
        # Prima era il centro-banda fisso del tipo; ora segue quanto stai andando davvero
        # (cadenza/velocità dai sensori). Se non c'è ancora segnale (cold start) ripiega sulla banda.
        live = live_entrainment_bpm(analysis)
        bpm = live if live is not None else (lo + hi) / 2
        tol = (hi - lo) / 2
        tau = params["tau"]
        genres = genres_for_mood(mood)
        weights = {"bpm": params["w_bpm"],
                   "energy": round(params["w_mood"] * 0.6, 3),
                   "valence": round(params["w_mood"] * 0.4, 3)}

    energy = params["energy"]

    # 3) fusione sensori: goal x effort
    if effort is not None:
        if goal == "EasyRun" and effort in ("HighEffort", "VeryHighEffort"):
            energy *= CALM
            bpm = lo + (bpm - lo) * 0.5                       # scendi verso banda bassa
        elif goal == "IntenseRun" and effort == "LowEffort":
            energy = min(1.0, energy * PUSH)
            bpm = bpm + (hi - bpm) * 0.5                      # sali verso banda alta
        elif goal == "ModerateRun":
            if effort == "LowEffort":
                energy = min(1.0, energy * 1.1); bpm = min(hi, bpm + 3)
            elif effort in ("HighEffort", "VeryHighEffort"):
                energy *= 0.9; bpm = max(lo, bpm - 3)
        # rifinitura trend (leggera): se sali ed sei gia' alto, non spingere oltre
        if trend == "Increasing" and effort in ("HighEffort", "VeryHighEffort"):
            energy = min(energy, params["energy"])

    # 4) variazione per tipo (ripetute, solo qualitativo): alterna veloce/lento ATTORNO al
    # bpm live (non piu' agli estremi fissi della banda), cosi' continua a seguire la velocita'.
    if goal == "IntenseRun" and last_bpm is not None and not quantitative:
        bpm = bpm - VAR_DELTA if last_bpm >= bpm else bpm + VAR_DELTA
        bpm = _clamp(bpm, ENTRAIN_MIN, ENTRAIN_MAX)          # resta nella cadenza naturale (Van Dyck)
        tol = NARROW                                         # punta preciso l'estremo scelto

    # 5) riscaldamento: nei primi minuti parti basso e sali fino al target previsto
    regime = "quantitative" if quantitative else "qualitative"
    if elapsed_min is not None and elapsed_min < WARMUP_MIN:
        f = max(0.0, elapsed_min) / WARMUP_MIN          # 0 -> parti da fermo, 1 -> a regime
        bpm = RECOVERY_BPM + (bpm - RECOVERY_BPM) * f
        energy = 0.20 + (energy - 0.20) * f
        regime = "warmup"

    # 6) riempimento + clamp
    valence = VALENCE_BY_MOOD.get(mood, 0.5)
    return Target(bpm=round(_clamp(bpm, 80, 200), 1),
                  energy=round(_clamp(energy, 0.0, 1.0), 3),
                  valence=valence, weights=weights, bpm_tolerance=round(tol, 1),
                  genres=genres, tau=round(tau, 3), mood=mood, goal=goal,
                  effort_band=GOAL_TO_EFFORT[goal], recovery=False, regime=regime)


def _demo() -> None:
    def show(titolo, t: Target):
        print(f"\n== {titolo} ==")
        print(f"  vettore   bpm={t.bpm} energy={t.energy} valence={t.valence}")
        print(f"  regime={t.regime}  tolleranza=±{t.bpm_tolerance}  tau={t.tau}  recovery={t.recovery}")
        print(f"  mood={t.mood} goal={t.goal} effort_band={t.effort_band}")
        print(f"  generi={'(tutti)' if not t.genres else str(len(t.genres)) + ' -> ' + ', '.join(t.genres[:6]) + '…'}")

    q = {"goal": "IntenseRun", "mood": "Energetic", "numbers": {"speed_kmh": 12},
         "target_bpm": 169, "params": GOAL_PARAMS["IntenseRun"]}
    ql = {"goal": "IntenseRun", "mood": "Energetic", "numbers": {},
          "target_bpm": None, "params": GOAL_PARAMS["IntenseRun"]}
    easy = {"goal": "EasyRun", "mood": "Calm", "numbers": {},
            "target_bpm": None, "params": GOAL_PARAMS["EasyRun"]}

    show("cold start quantitativo: 'ripetute a 12 km/h, carico'", decide(q))
    show("qualitativo: 'sono carico' (no numeri)", decide(ql))
    show("EasyRun ma effort VeryHigh (calmati)",
         decide(easy, analysis={"mean_hrr": 0.6, "effort_state": "VeryHighEffort", "trend_state": "Stable"}))
    show("IntenseRun qualitativo, ultima canzone veloce (varia -> lento)",
         decide(ql, analysis={"mean_hrr": 0.7, "effort_state": "TargetEffort", "trend_state": "Stable"}, last_bpm=182))
    show("SAFETY: HRR 0.95 -> recupero", decide(q, analysis={"mean_hrr": 0.95, "effort_state": "VeryHighEffort", "trend_state": "Increasing"}))

    print("\n== QUALITATIVO CHE INSEGUE LA VELOCITA' (piano -> spinge come un cavallo) ==")
    for v in (9, 11, 14, 17):
        t = decide(ql, analysis={"mean_hrr": 0.6, "effort_state": "TargetEffort",
                                 "trend_state": "Stable", "mean_speed_kmh": v})
        print(f"  velocità={v:>2} km/h  ->  bpm target={t.bpm}")

    print("\n== RISCALDAMENTO (quantitativo, elapsed 0->6 min): parti basso e sali ==")
    for e in (0, 1, 2, 3, 5, 6):
        t = decide(q, elapsed_min=e)
        print(f"  t={e}min  bpm={t.bpm:>5}  energy={t.energy:>5}  regime={t.regime}")


if __name__ == "__main__":
    _demo()

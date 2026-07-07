#  il seguente codice rappresnta il core del progetto: prende l'output del modulo NLP e lo 
#  stato corrente rilevato dai sensori, li fonde e costruisce il target che verrà passato al recommender



#  vengono utilizzate le librerie:
#    - annotations per usare annotazioni di tipo più flessibili
#    - dataclasses per creare classe target, convertire target in dizionario
#    - intent per recuperare parametri associati a (EasyRun/ModerateRun/IntenseRun), collegare goal a bande fisiologiche, stimare BPM da velocità
#    - genre_mood per recuperare generi compatibili con il mood
#    - symbolic per verificare tramite componente simbolica e ontologia se lo stato cardiaco è critico


from __future__ import annotations
from dataclasses import asdict, dataclass
from intent import GOAL_PARAMS, GOAL_TO_EFFORT, bpm_from_speed
from genre_mood import genres_for_mood
from symbolic import is_critical_state


#nel regime quantitativo si cercano canzoni entro una banda di +/- 5 BPM
NARROW = 5.0             

#nel reccomender la temperatura (tau) basse del softmax assegna probabilità molto alta a canzoni vicine. (explotation vs exploration)
TAU_EXPLOIT = 0.2

#quando utente chiede corsa facile ma sensori rilevano sforzo alto, l'energia viene moltiplicata per fattore 0.7
CALM = 0.7

#quando utente chiede corsa intensa ma sensori rilevano sforzo basso, l'energia viene moltiplicata per fattore 1.2
PUSH = 1.2

#durante recupero imposta BPM minimo dell'intervallo di BPM nell'allenamento EasyRun
RECOVERY_BPM = GOAL_PARAMS["EasyRun"]["bpm"][0]

#durata del warmup
WARMUP_MIN = 5.0         

#valori limite del BPM target
ENTRAIN_MIN, ENTRAIN_MAX = 150.0, 190.0

#nelle IntenseRun qualitative, il BPM target viene aumentato o diminuito di 8 (es. scatto-stop ripetuto)
VAR_DELTA = 8.0

#dizionario che mappa mood→target valence (secondo paper di Russell 1980 "arousal/positivita' della musica")
VALENCE_BY_MOOD = {"Energetic": 0.75, "Motivated": 0.70, "Focused": 0.45,
                   "Neutral": 0.50, "Calm": 0.25}



#classe contenente tutti i parametri necessari per il recommender

@dataclass
class Target:
    #3 parametri del target vector, vengono poi confrontati con le canzoni
    bpm: float
    energy: float
    valence: float

    #dizionario contenente il peso da associare ad ogni parametro del target vector
    weights: dict

    #valore che indica tolleranza tra BPM target e BPM canzone
    bpm_tolerance: float

    #generi ammessi
    genres: list

    #temperatura exploration/exploitation del softmax
    tau: float  

    mood: str
    goal: str
    effort_band: tuple       # classi matches_effort ammesse
    recovery: bool
    regime: str              # "quantitative" | "qualitative" | "recovery"


#  la seguente funzione converte il target nel vettore [BPM, energy, valence]
    def as_vector(self) -> list:
        return [self.bpm, self.energy, self.valence]

#  la seguente funzione converte il target in un dizionario
    def to_dict(self) -> dict:
        return asdict(self)


#  la seguente funzione serve per leggere un campo da PhysiologicalAnalysis
def _get(analysis, key, default=None):
    if analysis is None:
        return default
    if isinstance(analysis, dict):
        return analysis.get(key, default)
    return getattr(analysis, key, default)


#  la seguente funzione limita un valore a un intervallo
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


#  la seguente funzione indica il BPM che andrebbe seguito durante la corsa
#  per calcolarla si utilizza la cadenza misurata direttamente dal sensore
#  sennò la ricaviamo attraverso la velocità misurata

def live_entrainment_bpm(analysis) -> float | None:
    cadence = _get(analysis, "mean_cadence_spm")
    if cadence is not None and float(cadence) > 0:
        return _clamp(float(cadence), ENTRAIN_MIN, ENTRAIN_MAX)     # cadenza misurata, 1:1
    speed = _get(analysis, "mean_speed_kmh")
    if speed is not None and float(speed) > 0:
        return float(bpm_from_speed(float(speed)))                 # stima dalla velocità (Van Dyck)
    return None


#  funzione centrale, riceve: il dizionario intent contenente i parametri ricavati dalla frase dekll'utente in intent.py,
#  il dizionario analysis contenente i parametri rilevati dal sensore nel file physiological_state.py,
#  il BPM dell'ultima canzone riprodotto (last_bpm), i minuti trascorsi dall'inizio dell'allenamento (elapsed_min).
#  in base alla logica descritta genera il vettore target

def decide(intent: dict, analysis=None, last_bpm: float | None = None,
           elapsed_min: float | None = None) -> Target:

    #goal ricavato dalla frase tramite NLP
    goal = intent.get("goal") or "ModerateRun"

    #mood ricavato dalla frase tramite NLP
    mood = intent.get("mood") or "Neutral"

    #in base al goal si ricavano i parametri (banda BPM, energy, pesi, tau)
    params = intent.get("params") or GOAL_PARAMS[goal]
    
    #lower & upper bound dei BPM associati alla corsa desiderata
    lo, hi = params["bpm"]

    #BPM target ricavati dalla frase attreverso NLP
    target_bpm = intent.get("target_bpm")

    #se siamo riusciti a ricavare i BPM target, allora il regime è quantitativo
    quantitative = target_bpm is not None

    #recuperiamo parametri ricavati dallo stato fisiologico
    mean_hrr = _get(analysis, "mean_hrr")
    effort = _get(analysis, "effort_state")
    trend = _get(analysis, "trend_state")


    #  prima di procedere si effettua un safety override, ovvero si da priorità ai dati cardiaci
    #  viene passato l'HRR medio della finestra all'ontologia, se lo stato è critico si entra subito 
    #  in modalità recovery rallentando l'allenamento il più possibile

    if mean_hrr is not None and is_critical_state(mean_hrr):
        return Target(bpm=float(RECOVERY_BPM), energy=min(params["energy"], 0.30), valence=0.25,
                      weights={"bpm": 0.8, "energy": 0.15, "valence": 0.05},
                      bpm_tolerance=NARROW, genres=[], tau=TAU_EXPLOIT,
                      mood=mood, goal=goal, effort_band=("LowEffort", "TargetEffort"),
                      recovery=True, regime="recovery")

    
    #  se il regime è quantitativo si settano i seguenti parametri con i seguenti pesi
    if quantitative:
        bpm = float(target_bpm)
        tol = NARROW
        tau = TAU_EXPLOIT
        genres: list = []
        weights = {"bpm": 0.8, "energy": 0.15, "valence": 0.05}
    
    
    #  se invece è qualitativo il BPM segue la velocità reale del momento
    else:
        live = live_entrainment_bpm(analysis)
        bpm = live if live is not None else (lo + hi) / 2
        tol = (hi - lo) / 2
        tau = params["tau"]
        genres = genres_for_mood(mood)
        weights = {"bpm": params["w_bpm"],
                   "energy": round(params["w_mood"] * 0.6, 3),
                   "valence": round(params["w_mood"] * 0.4, 3)}

    #energy ricavata dai parametri legati al goal
    energy = params["energy"]

    # in questa sezione viene fatta la fusione tra goal ed effort
    # il target viene adattato allo stato reale del corpo

    if effort is not None:
        #se EasyRun ma sforzo alto, viene ridotta l'energy e spostato il BPM verso lower bound
        if goal == "EasyRun" and effort in ("HighEffort", "VeryHighEffort"):
            energy *= CALM
            bpm = lo + (bpm - lo) * 0.5                       
        
        #se IntenseRun ma sforzo basso, aumento energy e sposto BPM verso upper bound
        elif goal == "IntenseRun" and effort == "LowEffort":
            energy = min(1.0, energy * PUSH)
            bpm = bpm + (hi - bpm) * 0.5                      
        
        #se ModerateRun ma sforzo basso aumento enrgia e BPM
        #sennò se sforzo alto le abbasso
        elif goal == "ModerateRun":
            if effort == "LowEffort":
                energy = min(1.0, energy * 1.1); bpm = min(hi, bpm + 3)
            elif effort in ("HighEffort", "VeryHighEffort"):
                energy *= 0.9; bpm = max(lo, bpm - 3)
        
        #se trend Increasing e sforzo alto/molto alto, riporta l'energy al valore del goal
        if trend == "Increasing" and effort in ("HighEffort", "VeryHighEffort"):
            energy = min(energy, params["energy"])


    #se IntenseRun qualitativa, se BPM attuale è minore del BPM precedente, aumento BPM target, sennò diminuisco
    #l'obbiettivo è di alternare fase veloce-moderata
    if goal == "IntenseRun" and last_bpm is not None and not quantitative:
        bpm = bpm - VAR_DELTA if last_bpm >= bpm else bpm + VAR_DELTA
        bpm = _clamp(bpm, ENTRAIN_MIN, ENTRAIN_MAX)          
        tol = NARROW                                         

    #fase di riscaldamento: nei primi minuti parti basso e sali fino al target previsto
    #si specifica regime, si aumenta lentamente il BPM e l'energia
    regime = "quantitative" if quantitative else "qualitative"
    if elapsed_min is not None and elapsed_min < WARMUP_MIN:
        f = max(0.0, elapsed_min) / WARMUP_MIN
        bpm = RECOVERY_BPM + (bpm - RECOVERY_BPM) * f
        energy = 0.20 + (energy - 0.20) * f
        regime = "warmup"


    #nel warmup si accettano canzoni associate a LowEffort/TargetEffort anche se il goal fosse IntenseRun
    effort_band = ("LowEffort", "TargetEffort") if regime == "warmup" else GOAL_TO_EFFORT[goal]
    valence = VALENCE_BY_MOOD.get(mood, 0.5)
    
    #creazione target finale
    return Target(bpm=round(_clamp(bpm, 80, 200), 1),
                  energy=round(_clamp(energy, 0.0, 1.0), 3),
                  valence=valence, weights=weights, bpm_tolerance=round(tol, 1),
                  genres=genres, tau=round(tau, 3), mood=mood, goal=goal,
                  effort_band=effort_band, recovery=False, regime=regime)






# simulazione
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

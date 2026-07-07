# RUNMAXXIN

Sistema che raccomanda canzoni durante una corsa. L'utente scrive un prompt breve
(≤20 parole); durante l'allenamento arrivano i dati dei sensori; il sistema decide,
canzone dopo canzone, che musica proporre. Questo documento spiega ogni componente.

Il codice è pensato per un esame di AI-Lab: NLP + ontologia + logica sui sensori +
un sistema di raccomandazione (quest'ultimo sviluppato da un collega).

## La pipeline

```
prompt ──► [1] NLP ──► intento (goal, mood, numeri, bpm)
                                  │
dati sensori ──► [2] sensori ──► stato (HRR, sforzo, trend)
                                  │
                        [controller] fonde intento + sensori ──► vettore TARGET
                                  │
                        [3] recommender (collega): distanza + softmax ──► prossima canzone
```

Due "regimi": **quantitativo** (l'utente dichiara una velocità → BPM preciso) e
**qualitativo** (nessun numero → banda di BPM + generi coerenti col mood).

## I componenti, pezzo per pezzo

### Stadio 0 — Dati e sensori di base (forniti dal team)
- **`physiological_state.py`** — data una finestra di battiti (BPM), calcola: HRR di Karvonen
  `(HR−HR_rest)/(HR_max−HR_rest)`, la classe di sforzo (`effort_state` ∈ Low/Target/High/VeryHigh),
  e il trend (regressione lineare dei BPM → Increasing/Decreasing/Stable). Gestisce anche
  velocità e cadenza.
- **`build_dataset.py`** — prende un CSV di sessioni (una riga al secondo) e produce
  `physiological_windows.csv`: una riga per **finestra da 30 secondi**, con le feature di sopra.
- **`songs.csv`** — catalogo di ~89.000 canzoni con `bpm, energy, valence, danceability, genre`
  e le etichette `supports_mood`, `supports_goal`, `matches_effort`.

### Stadio 1 — NLP: dal testo alle feature (`intent.py`)
`route(frase)` restituisce `{goal, mood, numbers, target_bpm, params}`:
- **numeri** (velocità, passo, distanza, durata): estratti con regex.
- **BPM target**: se c'è una velocità, `bpm_from_speed` la converte in cadenza→BPM
  (cadenza = 134 + 2.9·velocità, limitata a 150–190; entrainment 1:1).
- **goal** (EasyRun/ModerateRun/IntenseRun): prima un **override a parole-chiave** deterministico
  (es. "ripetute", "spingere", "recupero", "maratona"); se nessuna parola-chiave, un
  classificatore **SetFit** (few-shot).
- **mood** (Neutral/Focused/Energetic/Motivated/Calm): classificatore **SetFit**.
- **`train_intent.py`** addestra i due modelli SetFit dai loro esempi few-shot. I modelli
  (~449 MB l'uno) NON sono nel repo (gitignorati): si rigenerano con `python train_intent.py`.

### Ontologia mood → generi (`ontology/genre_mood.owl`, `genre_mood.py`)
Ontologia OWL (classi `Mood`/`Genre`, proprietà `genreSuitsMood`/`dominantMood`, individui).
Fornisce, dato un mood, i generi candidati per il regime qualitativo. È **knowledge-driven**:
le associazioni mood↔genere sono definite **a priori dalla teoria** (Russell 1980, valenza ×
arousal; Karageorghis & Terry 2009, arousal per famiglia di generi), non dedotte dai dati. Il
catalogo `songs.csv` popola **solo le istanze** (quali generi esistono, A-Box) e può validare
l'ontologia; non definisce le regole (T-Box). `build_mood_ontology.py` rigenera il file;
`genre_mood.py` lo interroga (`genres_for_mood`).

### Controller — genera il vettore target (`controller.py`)
`decide(intent, analysis, last_bpm, elapsed_min)` produce un `Target` = vettore
`[bpm, energy, valence]` + `weights`, `bpm_tolerance`, `genres`, `tau`, `effort_band`, `recovery`.
Logica, in ordine di priorità:
1. **Safety**: se `mean_hrr ≥ 0.90`, forza un vettore di recupero (BPM basso, energia bassa) —
   vince su tutto.
2. **Regime**: quantitativo → BPM preciso, raggio stretto, `tau` basso; qualitativo → banda,
   raggio largo, generi del mood, `tau` più alto (più varietà).
3. **Fusione sensori**: aggiusta energia/BPM in base allo sforzo misurato (es. "dici easy ma
   sei in affanno" → abbassa).
4. **Riscaldamento**: nei primi 5 minuti il target sale gradualmente da ~120 BPM al valore previsto.
5. **Variazione**: su IntenseRun alterna veloce/lento rispetto alla canzone precedente.
Il contratto completo del `Target` è in `docs/TARGET_CONTRACT.md`.

### Loop di sessione (`session.py`)
Concatena gli stadi: calcola l'intento una volta, poi legge le finestre da 30s di
`build_dataset`, le raggruppa per canzone, chiama `decide(...)` e passa il target al recommender.
Il recommender è qui uno **stub** (canzone col BPM più vicino) — un gancio per il collega.
`main.py` è una versione minimale (prompt → intento → target, senza loop).

### Stadio 3 — Recommender (del collega, NON incluso)
Riceve il `Target`, calcola una distanza pesata dai vettori-canzone e assegna probabilità con
un softmax (`P ∝ exp(−distanza/tau)`). Non è parte di questo repo; il contratto d'interfaccia è
documentato in `docs/TARGET_CONTRACT.md`.

### Dati simulati (`simulate_sessions.py`)
Genera `data/simulated/bpm_sessions.csv` con due sessioni da 30 minuti (input per `build_dataset`):
- `marathon_ontarget` — atleta a ritmo costante, resta in zona target, non si affatica;
- `push_then_fatigue` — parte forte poi si affatica: il cuore sale mentre la velocità cala.
Sono **dati sintetici**, servono a esercitare la pipeline in assenza di sensori reali.

### Valutazione NLP (`eval_intent.py`)
Su un insieme di frasi tenute fuori dal training misura accuracy e F1 per classe dei
classificatori SetFit, e li confronta con il baseline a parole-chiave (ablation).

### Test (`test_intent.py`, `test_genre_mood.py`)
Test deterministici (non caricano i modelli): regex dei numeri, `bpm_from_speed`, coerenza
delle etichette col catalogo, override a parole-chiave, integrità dell'ontologia.

## Come si esegue
```bash
pip install -r requirements.txt
python train_intent.py          # genera i modelli SetFit (in models/, gitignorati)
python simulate_sessions.py     # genera i dati sensori simulati
python build_dataset.py --input data/simulated/bpm_sessions.csv \
                        --output data/processed/physiological_windows.csv
python session.py "oggi voglio spingere tantissimo" --session push_then_fatigue
python -m pytest -q             # test deterministici
python eval_intent.py           # metriche NLP (richiede i modelli)
```

## Fondamenti teorici
Ogni formula/scelta con la sua fonte è in **`docs/THEORY.md`** (Karvonen 1957 per l'HRR;
Tanaka 2001 per HR_max; Van Dyck 2015 per cadenza↔tempo musicale; Russell 1980 e
Karageorghis & Terry 2009 per mood/arousal; Sutton & Barto per softmax/exploration;
Rada 1989 per l'ontologia). Le scelte di design (soglie, pesi, τ) sono dichiarate come tali.

## Stato del progetto
| Componente | Stato |
|---|---|
| Sensori (`physiological_state`, `build_dataset`) | forniti dal team |
| NLP (`intent`, training, valutazione) | fatto |
| Ontologia mood→generi | fatto |
| Controller (vettore target) | fatto |
| Loop di sessione | fatto (con recommender stub) |
| Recommendation system | **da fare — sviluppato dal collega** |

## Note oneste (limiti)
- I modelli SetFit sono few-shot: coprono le frasi tipiche ma possono sbagliare su formulazioni
  fuori distribuzione (per questo c'è l'override a parole-chiave e, a valle, la correzione dei sensori).
- I dati delle sessioni sono **simulati**, non misurati.
- L'ontologia è **knowledge-driven** (teoria di Russell): essendo indipendente dalle etichette
  `supports_mood`, queste possono essere usate come **test set** per validarla (accordo osservato).
- La velocità NON determina il tipo di allenamento (è relativa alla forma dell'atleta): lo sforzo
  reale lo misurano i sensori.

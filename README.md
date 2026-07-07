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

### Mood → generi: dizionario (`genre_mood.py`)
Era un file OWL letto a regex — funzionalmente già un dizionario (nessuna query, nessuna
inferenza), solo scritto in un formato più complicato: ora è dichiarato per quello che è. La
**regola** resta **knowledge-driven**: le associazioni mood↔genere sono definite **a priori
dalla teoria** (Russell 1980, valenza × arousal; Karageorghis & Terry 2009, arousal per famiglia
di generi), non dedotte dai dati. `songs.csv` fornisce solo l'**elenco dei generi esistenti**;
il dizionario (`GENRE_TO_MOODS`) è calcolato una volta all'import, interrogabile con
`genres_for_mood(mood)`.

### Controller — genera il vettore target (`controller.py`)
`decide(intent, analysis, last_bpm, elapsed_min)` produce un `Target` = vettore
`[bpm, energy, valence]` + `weights`, `bpm_tolerance`, `genres`, `tau`, `effort_band`, `recovery`.
Logica, in ordine di priorità:
1. **Safety**: se lo stato è `CriticalState` (vedi sotto), forza un vettore di recupero
   (BPM basso, energia bassa) — vince su tutto.
2. **Regime**: quantitativo → BPM preciso, raggio stretto, `tau` basso; qualitativo → banda,
   raggio largo, generi del mood, `tau` più alto (più varietà).
3. **Fusione sensori**: aggiusta energia/BPM in base allo sforzo misurato (es. "dici easy ma
   sei in affanno" → abbassa).
4. **Riscaldamento**: nei primi 5 minuti il target sale gradualmente da ~120 BPM al valore previsto.
5. **Variazione**: su IntenseRun alterna veloce/lento rispetto alla canzone precedente.
Il contratto completo del `Target` è in `docs/TARGET_CONTRACT.md`.

### Strato simbolico (`symbolic.py`, `ontology/runner_state.owl`, `ontology/nlp_shapes.ttl`)
Due responsabilità reali dell'ontologia (non un lookup, non un `if` in Python):
- **`is_critical_state(mean_hrr)`**: la soglia di sicurezza (0.90) è un **dato dichiarato
  nell'ontologia** (`ar:CriticalState ar:hasThreshold`), non una costante Python. Si inietta la
  HRR osservata come tripla RDF e una **query SPARQL** (via `rdflib`) confronta i due valori
  dentro il grafo — è la query a classificare lo stato, non un confronto scritto a mano. Nessun
  reasoner DL/Java: solo `rdflib`, deterministico.
- **`validate_speed(speed_kmh)`**: una **SHACL shape** (`pyshacl`) valida i numeri estratti
  dall'NLP contro un vincolo fisiologico (0–45 km/h) — pattern "Constraint Gate": l'ontologia
  non genera dati, li restringe e valida. Una velocità assurda ("300 km/h") viene scartata
  esplicitamente invece di essere silenziosamente clampata più a valle.
- **`is_effort_compatible(song_efforts, effort_band)`**: pattern **Generator→Validator**. Il
  recommender ottimizza `[bpm,energy,valence]` ma ignora la compatibilità di sforzo — misurato:
  il Top-1 viola l'`effort_band` nel **16.7% dei casi (20/120)** su un campione ampio. Una query
  SPARQL verifica l'intersezione fra `matches_effort` della canzone e `effort_band` del target;
  `session.pick_valid_song` scorre le Top-K del recommender e sceglie la prima compatibile.
Il resto della logica del controller (riscaldamento, fusione, variazione, clamp) resta calcolo
numerico in Python: OWL/SWRL non sono adatti al calcolo continuo, solo alla classificazione.

### Loop di sessione (`session.py`)
Concatena gli stadi: calcola l'intento una volta, poi legge le finestre da 30s di
`build_dataset`, le raggruppa per canzone, chiama `decide(...)`, passa il target al
**recommender del collega** (Top-K per distanza+softmax) e applica l'**effort gate** simbolico
per scegliere la canzone finale. `main.py` è una versione minimale (prompt → intento → target →
Top-K, senza loop né gate).

### Stadio 3 — Recommender (`recommender.py`)
Riceve il `Target`, calcola la distanza euclidea pesata dai vettori-canzone (BPM normalizzato
da `bpm_tolerance`) e assegna probabilità con un softmax (`P ∝ exp(−distanza/τ)`); restituisce
le Top-K canzoni, escludendo le tracce recenti e filtrando sui generi del mood nel regime
qualitativo. Consuma esattamente il `Target` del controller (contratto in
`docs/TARGET_CONTRACT.md`). Implementato in numpy/pandas.

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
| Mood → generi (`genre_mood.py`, dizionario knowledge-driven) | fatto |
| Controller (vettore target) | fatto |
| Strato simbolico (`symbolic.py`: safety SPARQL, SHACL NLP, effort gate) | fatto |
| Loop di sessione (`session.py`, `main.py`) | fatto (recommender + gate) |
| Recommendation system (`recommender.py`) | fatto (distanza pesata + softmax) |

## Note oneste (limiti)
- I modelli SetFit sono few-shot: coprono le frasi tipiche ma possono sbagliare su formulazioni
  fuori distribuzione (per questo c'è l'override a parole-chiave e, a valle, la correzione dei sensori).
- I dati delle sessioni sono **simulati**, non misurati.
- Il dizionario mood→generi è **knowledge-driven** (teoria di Russell): essendo indipendente
  dalle etichette `supports_mood`, queste possono essere usate come **test set** per validarlo
  (accordo osservato), non per costruirlo.
- La velocità NON determina il tipo di allenamento (è relativa alla forma dell'atleta): lo sforzo
  reale lo misurano i sensori.
- Il layer simbolico (SPARQL/SHACL) classifica e valida **stati discreti** (soglie, insiemi
  compatibili); non gestisce calcolo numerico continuo (riscaldamento, fusione, softmax), che
  resta in Python — non è una limitazione di implementazione ma una scelta di dove il
  ragionamento simbolico è appropriato.

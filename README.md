# RUNMAXXIN

Sistema che raccomanda canzoni durante una corsa. L'utente scrive un prompt breve
(≤20 parole); durante l'allenamento arrivano i dati dei sensori; il sistema decide,
canzone dopo canzone, che musica proporre. Questo documento spiega ogni componente.

Il codice è pensato per un esame di AI-Lab: NLP + strato simbolico + logica sui sensori +
un sistema di raccomandazione (quest'ultimo sviluppato da un collega).

## La pipeline

```
prompt ──► [1] NLP ──► intento (goal, mood, numeri, bpm)
                                  │
dati sensori ──► [2] sensori ──► stato (HRR, sforzo, trend, velocità, cadenza)
                                  │
                        [controller] fonde intento + sensori ──► vettore TARGET
                                  │
             [3] recommender (collega): distanza + softmax ──► Top-K candidati
                                  │
             [gate simbolico] dedup + compatibilità sforzo + campionamento ──► canzone
```

Due "regimi": **quantitativo** (l'utente dichiara una velocità → BPM preciso e fisso) e
**qualitativo** (nessun numero → il BPM **insegue la velocità reale** misurata dai sensori,
entro la banda del tipo/mood).

## I componenti, pezzo per pezzo

### Stadio 0 — Dati e sensori di base (forniti dal team)
- **`physiological_state.py`** — data una finestra di battiti (BPM), calcola: HRR di Karvonen
  `(HR−HR_rest)/(HR_max−HR_rest)`, la classe di sforzo (`effort_state` ∈ Low/Target/High/VeryHigh),
  e il trend (regressione lineare dei BPM → Increasing/Decreasing/Stable). Gestisce anche
  velocità e cadenza.
- **`build_dataset.py`** — prende un CSV di sessioni (una riga al secondo) e produce
  `physiological_windows.csv`: una riga per **finestra da 30 secondi**, con le feature di sopra.
- **`songs.csv`** — catalogo di ~89.503 canzoni (dati Spotify) con `bpm, energy, valence,
  danceability, genre` e le etichette `supports_mood`, `supports_goal`, `matches_effort`.
  Contiene **duplicati reali**: la stessa traccia può comparire con `song_id` diversi (fino
  a ~45 copie per alcuni brani) — gestito a valle (vedi "Loop di sessione").

### Stadio 1 — NLP: dal testo alle feature (`intent.py`)
`route(frase)` restituisce `{goal, mood, numbers, target_bpm, params}`:
- **numeri** (velocità, passo, distanza, durata): estratti con regex.
- **BPM target**: se c'è una velocità, `bpm_from_speed` la converte in cadenza→BPM
  (cadenza = 134 + 2.9·velocità, limitata a 150–190; entrainment 1:1, Van Dyck 2015).
- **goal** (EasyRun/ModerateRun/IntenseRun): prima un **override a parole-chiave** deterministico
  (es. "ripetute", "spingere", "recupero", "maratona"); se nessuna parola-chiave, un
  classificatore **SetFit** (few-shot).
- **mood** (Neutral/Focused/Energetic/Motivated/Calm): classificatore **SetFit**.
- **`train_intent.py`** addestra i due modelli SetFit dai loro esempi few-shot. I modelli
  (~449 MB l'uno) NON sono nel repo (gitignorati): si rigenerano con `python train_intent.py`.
- **Limite noto**: i modelli sono few-shot, addestrati su poche decine di frasi per classe —
  su formulazioni fuori dai loro esempi possono sbagliare (es. classificare sempre la classe
  più frequente). L'override a parole-chiave in `route()` copre i casi più comuni; per il resto
  serve un dataset di training più ampio e bilanciato (non ancora fatto).

### Mood → generi: dizionario (`genre_mood.py`)
Un dizionario Python (`GENRE_TO_MOODS`, calcolato una volta all'import). Le associazioni
mood↔genere sono **knowledge-driven**: derivano a priori dalla teoria (Russell 1980, valenza ×
arousal; Karageorghis & Terry 2009, arousal per famiglia di generi), non dedotte dai dati.
`songs.csv` fornisce solo l'**elenco dei generi esistenti**. Si interroga con
`genres_for_mood(mood)`.

### Controller — genera il vettore target (`controller.py`)
`decide(intent, analysis, last_bpm, elapsed_min)` produce un `Target` = vettore
`[bpm, energy, valence]` + `weights`, `bpm_tolerance`, `genres`, `tau`, `effort_band`, `recovery`.
Logica, in ordine di priorità:
1. **Safety**: se lo stato è critico (vedi "Strato simbolico"), forza un vettore di recupero
   (BPM basso, energia bassa) — vince su tutto.
2. **Regime**:
   - **quantitativo** (velocità dichiarata nel prompt): BPM preciso e fisso, raggio stretto,
     `tau` basso (exploit).
   - **qualitativo** (nessun numero): il BPM **insegue la velocità/cadenza reale del momento**
     (`live_entrainment_bpm`, entrainment continuo di Van Dyck) — se manca ancora un segnale
     (cold start) ripiega sul centro-banda del tipo. Raggio largo, generi del mood, `tau` più
     alto (explore).
3. **Fusione sensori**: aggiusta energia/BPM in base allo sforzo misurato (es. "dici easy ma
   sei in affanno" → abbassa; "dici intense ma vai piano" → alza).
4. **Variazione**: su IntenseRun alterna veloce/lento rispetto alla canzone precedente,
   attorno al BPM live (non più agli estremi fissi della banda).
5. **Riscaldamento**: nei primi 5 minuti il target sale gradualmente da ~120 BPM al valore
   previsto. In questa fase l'`effort_band` ammessa è rilassata (Low/TargetEffort) e non quella
   nominale del tipo — altrimenti il gate di sforzo non troverebbe mai canzoni compatibili
   mentre la musica è ancora calma.
Il contratto completo del `Target` è in `docs/TARGET_CONTRACT.md`.

### Strato simbolico (`symbolic.py`, `ontology/runner_state.owl`, `ontology/nlp_shapes.ttl`)
Tre responsabilità reali (non un lookup, non un `if` scritto a mano in mezzo alla logica):
- **`is_critical_state(mean_hrr)`**: la soglia di sicurezza (0.90) è un **dato dichiarato
  nell'ontologia** (`ar:CriticalState ar:hasThreshold`), non una costante Python. Si inietta la
  HRR osservata come tripla RDF e una **query SPARQL** (via `rdflib`) confronta i due valori
  dentro il grafo — è la query a classificare lo stato. Nessun reasoner DL/Java: solo `rdflib`,
  deterministico.
- **`validate_speed(speed_kmh)`**: una **SHACL shape** (`pyshacl`) valida i numeri estratti
  dall'NLP contro un vincolo fisiologico (0–45 km/h) — pattern "Constraint Gate": l'ontologia
  non genera dati, li restringe e valida. Una velocità assurda ("300 km/h") viene scartata
  esplicitamente invece di essere silenziosamente clampata più a valle.
- **`is_effort_compatible(song_efforts, effort_band)`**: pattern **Generator→Validator**. Il
  recommender ottimizza `[bpm,energy,valence]` ma ignora la compatibilità di sforzo — misurato
  su un campione ampio: il Top-1 viola l'`effort_band` nel **16.7% dei casi (20/120)**. Una
  query SPARQL verifica l'intersezione fra `matches_effort` della canzone e `effort_band` del
  target; `session.pick_song` la usa per filtrare i candidati (vedi sotto).
Il resto della logica del controller (riscaldamento, fusione, variazione, clamp) resta calcolo
numerico in Python: OWL/SWRL non sono adatti al calcolo continuo, solo alla classificazione.

### Stadio 3 — Recommender (`recommender.py`, del collega)
Riceve il `Target`, calcola la distanza euclidea pesata dai vettori-canzone (BPM normalizzato
da `bpm_tolerance`) e assegna probabilità con un softmax (`P ∝ exp(−distanza/τ)`); restituisce
le Top-K canzoni ordinate, escludendo gli id già esclusi e filtrando sui generi del mood nel
regime qualitativo. Consuma esattamente il `Target` del controller (contratto in
`docs/TARGET_CONTRACT.md`). Implementato in numpy/pandas.
- **Limite noto (misurato)**: il recommender calcola il softmax ma poi **ordina e prende il
  primo** — è quindi deterministico: target simili restituiscono sempre le stesse canzoni,
  `tau` non ha alcun effetto pratico. Il correttivo vive a valle, in `session.pick_song`.

### Gate a valle sul Top-K (`session.pick_song`, usato da `session.py`, `run_demo.py`, `tester.py`)
Tre correzioni applicate DOPO il recommender, senza toccare il suo modulo:
1. **Deduplica** le Top-K per (titolo, artista) — il catalogo ha copie della stessa traccia
   con `song_id` diversi (`load_song_variants`); altrimenti la "stessa" canzone può ripresentarsi
   sotto un id diverso.
2. **Filtra** con `is_effort_compatible` (il gate simbolico sopra).
3. **Campiona** (non prende sempre il primo) fra le candidate compatibili pesando per la
   `probability` del recommender — softmax/Boltzmann (Sutton & Barto): con un seed per sessione
   resta riproducibile, ma sessioni diverse danno playlist diverse (prima, essendo il recommender
   deterministico, sessioni diverse con target simili restituivano le stesse identiche canzoni).
Quando una canzone viene scelta, **tutte le sue varianti** (stesso titolo+artista, id diversi)
vengono escluse dai turni successivi, non solo quel singolo `song_id`.

### Loop di sessione (`session.py`)
Concatena gli stadi leggendo una sessione **pre-registrata**: calcola l'intento una volta, legge
le finestre da 30s di `build_dataset`, le raggruppa per canzone, chiama `decide(...)`, passa il
target a `recommender.recommend` (Top-K) e applica `pick_song` per la scelta finale.
`main.py` è una versione minimale a freddo (prompt → intento → target → Top-K, senza loop).

## Demo e test (gli entrypoint da guardare)

### Tester — sessioni generate al volo (`tester.py`)
```bash
pip install -r requirements.txt
python tester.py                # prompt in inglese, poi scegli una prestazione; play 6s/canzone
python tester.py --seed 42      # riproduci una run specifica (default: casuale ogni volta)
```
UI in inglese (l'esame è in inglese). Non replica un CSV fisso: **genera** la sessione in
memoria, secondo tre fattori indipendenti:
- **prompt** → intento (goal/mood) **e scala**: se dichiara una distanza ("20 km") o una
  durata ("40 min") la sessione dura quello; se dichiara un passo ("run at 12 km/h") entra in
  regime **quantitativo** e quel passo fissa anche il **livello** di velocità generato (la
  forma dell'archetipo viene riscalata su quel livello); senza numeri, durata **random**
  (18–42 min) e regime **qualitativo**.
- **archetipo scelto** (uno dei 6 di `simulate_sessions.py`) → la *forma* dell'andamento
  (steady, in salita, ripetute, …), non valori fissi.
- **rumore stocastico** → battito/velocità generati con variazione casuale attorno alla forma:
  ogni run è diversa (`--seed` per riprodurne una).

I valori generati passano dal `physiological_state.py` di base (HRR → sforzo/trend), poi dal
controller e dal recommender+gate. Ogni canzone dura la **sua** durata reale (stimata in
`data/song_durations.csv`, generata la prima volta perché `songs.csv` non contiene la durata —
vedi "Note oneste"), non un blocco a tempo fisso. Il tester mostra l'andamento (cuore+velocità),
i sensori, il target, e **solo la canzone scelta** (niente più pannello Top-3: con la scelta
stocastica, mostrare "le 3 più vicine" non avrebbe più senso). A fine allenamento stampa un
**WORKOUT SUMMARY** (durata, HRR/velocità media e di picco, mix di sforzo, canzoni di recupero)
e la **PLAYLIST** completa (ogni canzone con orario, artista, genere, BPM, durata, regime).

### Replay di una sessione pre-registrata, non interattivo (`run_demo.py`)
```bash
python run_demo.py "oggi voglio spingere tantissimo" --session push_fatigue
```
Cruscotto animato in italiano che scorre una sessione fissa (da `simulate_sessions.py` +
`build_dataset.py`) canzone per canzone. ⚠️ Il valore di default di `--session` nello script
(`push_then_fatigue`) **non corrisponde più** ai nomi delle sessioni generate oggi
(`push_fatigue`, vedi sotto) — va sempre passato esplicitamente `--session <nome valido>` finché
questo non viene sistemato nel codice.

### Sessioni pre-registrate (`simulate_sessions.py`)
Genera `data/simulated/bpm_sessions.csv` con **6 archetipi** di prestazione, durate e dinamiche
diverse (input per `build_dataset.py`, usati da `session.py`/`run_demo.py`/`tester.py`):
`steady` (35 min, ritmo costante) · `push_fatigue` (30 min, spinge poi cede) ·
`negative_split` (40 min, parte piano e accelera) · `intervals` (25 min, ripetute) ·
`easy_recovery` (20 min, sforzo basso costante) · `beginner_struggle` (22 min, erratico,
cuore alto anche a bassa velocità, pause camminata). Sono **dati sintetici**.

### Valutazione NLP (`eval_intent.py`)
Su un insieme di frasi tenute fuori dal training misura accuracy e F1 per classe dei
classificatori SetFit, e li confronta con il baseline a parole-chiave (ablation).

### Test automatici (`test_intent.py`, `test_genre_mood.py`, `test_symbolic.py`)
Test deterministici (non caricano i modelli SetFit): regex dei numeri, `bpm_from_speed`,
coerenza delle etichette col catalogo, override a parole-chiave, integrità del dizionario
mood→generi, classificazione SPARQL del safety-state, validazione SHACL.

## Come si esegue (pipeline completa, sessione fissa)
```bash
pip install -r requirements.txt
python train_intent.py          # genera i modelli SetFit (in models/, gitignorati)
python simulate_sessions.py     # genera i 6 archetipi di sessione simulata
python build_dataset.py --input data/simulated/bpm_sessions.csv \
                        --output data/processed/physiological_windows.csv
python session.py "oggi voglio spingere tantissimo" --session push_fatigue
python -m pytest -q             # test deterministici
python eval_intent.py           # metriche NLP (richiede i modelli)
```
Per il tester generativo basta `python tester.py` — non serve `build_dataset`/`simulate_sessions`
a monte (la sessione la costruisce lui).

## Fondamenti teorici
Ogni formula/scelta con la sua fonte è in **`docs/THEORY.md`** (Karvonen 1957 per l'HRR;
Tanaka 2001 per HR_max; Van Dyck 2015 per cadenza↔tempo musicale, anche continua; Russell 1980 e
Karageorghis & Terry 2009 per mood/arousal; Sutton & Barto per softmax/exploration;
Rada 1989 per la struttura a rete dei generi). Le scelte di design (soglie, pesi, τ) sono
dichiarate come tali.

## Stato del progetto
| Componente | Stato |
|---|---|
| Sensori (`physiological_state`, `build_dataset`) | forniti dal team |
| NLP (`intent`, training, valutazione) | fatto, **limite noto**: pochi dati few-shot |
| Mood → generi (`genre_mood.py`, dizionario knowledge-driven) | fatto |
| Controller (vettore target, entrainment continuo) | fatto |
| Strato simbolico (`symbolic.py`: safety SPARQL, SHACL NLP, effort gate) | fatto |
| Gate a valle (dedup + sampling, `session.pick_song`) | fatto |
| Loop di sessione fissa (`session.py`, `main.py`, `run_demo.py`) | fatto (`run_demo.py` ha un default di sessione disallineato) |
| Tester generativo (`tester.py`) | fatto |
| Recommendation system (`recommender.py`, del collega) | fatto, **limite noto**: deterministico (softmax calcolato ma non usato per scegliere) |

## Note oneste (limiti)
- I modelli SetFit sono few-shot su un numero ridotto di esempi: coprono le frasi tipiche ma
  possono sbagliare su formulazioni fuori distribuzione (l'override a parole-chiave e, a valle,
  la correzione dei sensori mitigano ma non risolvono). Non ancora riaddestrati con un dataset
  più ampio e bilanciato.
- I dati delle sessioni sono **simulati**, non misurati su una corsa reale.
- Il dizionario mood→generi è **knowledge-driven** (teoria di Russell): essendo indipendente
  dalle etichette `supports_mood` del catalogo, queste possono essere usate come **test set**
  per validarlo (accordo osservato), non per costruirlo.
- La velocità NON determina da sola il tipo di allenamento dichiarato (è relativa alla forma
  dell'atleta): lo sforzo reale lo misurano i sensori, non un numero di velocità isolato.
- Il layer simbolico (SPARQL/SHACL) classifica e valida **stati discreti** (soglie, insiemi
  compatibili); non gestisce calcolo numerico continuo (riscaldamento, fusione, entrainment),
  che resta in Python — scelta di dove il ragionamento simbolico è appropriato, non una
  limitazione di implementazione.
- `songs.csv` non ha una colonna di durata: `tester.py` ne genera una deterministica per
  brano (`data/song_durations.csv`, 2:30–4:00) invece di usare valori reali (non disponibili
  senza chiamare l'API di Spotify).
- `songs.csv` contiene brani duplicati (stesso titolo/artista, `song_id` diversi, fino a ~45
  copie); gestito a valle in `session.pick_song`/`load_song_variants`, non nel catalogo stesso.
- Il recommender (`recommender.py`) è **deterministico** nonostante calcoli un softmax — non
  usa `tau` per scegliere, solo per assegnare una probabilità che poi viene ordinata; la
  variabilità reale (exploration) è stata aggiunta a valle in `session.pick_song`.
- `run_demo.py` ha un valore di default per `--session` (`push_then_fatigue`) che non
  corrisponde più ai nomi generati da `simulate_sessions.py` (`push_fatigue`) dopo l'aggiornamento
  agli archetipi multipli — va passato esplicitamente.
- Nel repository sono presenti anche `intent_EN.py` e `intent_IT.py`, caricati separatamente
  e non usati da nessun altro modulo: da riconciliare con `intent.py` (l'unico effettivamente
  in uso nella pipeline) prima della consegna, per evitare confusione su quale sia la versione
  di riferimento.

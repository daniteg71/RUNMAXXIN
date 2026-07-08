# THEORY.md — Fondamenti teorici e citazioni (RUNMAXXIN)

Ogni formula/scelta con la sua **fonte**. Regola: ciò che ha una fonte si cita; ciò che è
**scelta di design** si dichiara come tale e si giustifica con un'**ablation** — mai citazioni
finte. Legenda: ✅ = fonte citabile · ⚙️ = design (→ ablation).

## Stadio 1 — NLP: frase → features (`intent.py`, `train_intent.py`)

### N1. Riconoscimento del tipo di allenamento e del mood (regole + SetFit)
Due classificatori few-shot (SetFit) sul vocabolario di `songs.csv`:
`goal ∈ {EasyRun, ModerateRun, IntenseRun}` e `mood ∈ {Neutral, Focused, Energetic, Motivated, Calm}`.
Il `goal` usa un **override a parole-chiave** deterministico quando la frase contiene un segnale
inequivocabile (es. "ripetute", "spingere", "recupero", "maratona"); altrimenti decide SetFit.
Lo stesso override è il **baseline** del confronto in `eval_intent.py` (SetFit vs regole → ablation
con accuracy e F1 per classe su un test tenuto fuori). La velocità NON determina il goal (è
relativa alla forma dell'atleta): lo sforzo reale lo misura il sensore (Karvonen).
- ✅ **Russell 1980** — modello circumplex (valenza × arousal): il mood vive in uno spazio 2D
  che orienta la scelta musicale.
- ✅ **Karageorghis & Terry 2009** — tempo/energia musicale ↔ arousal e resa sportiva.
- ⚙️ set di label, frasi few-shot e l'uso di SetFit = design → ablation.

**Risultati eval (`eval_intent.py`, test tenuto fuori):** MOOD accuracy 1.000 (10/10). GOAL
accuracy 0.833 (10/12) contro 0.750 del baseline a keyword — SetFit generalizza meglio ma non
è perfetto. La confusion matrix isola l'errore: **2 frasi IntenseRun senza gergo tecnico**
("cuore in gola a fasi alterne con scatti", "do' tutto me stesso fino a crollare") vengono
classificate **EasyRun**, mai ModerateRun — non è rumore ma un bias sistematico del training
set (`GOAL_TRAIN`), che ha pochi esempi IntenseRun "senza gergo" rispetto a EasyRun/ModerateRun
"senza gergo": il classificatore associa l'assenza di lessico tecnico a bassa intensità invece
di leggere la semantica dello sforzo massimo. Fix proposto: arricchire `GOAL_TRAIN` con altri
esempi IntenseRun senza gergo prima di ripetere `train_intent.py`.

### N2. Numeri dalla frase → BPM desiderato (regime quantitativo)   `bpm_from_speed`
```
cadenza(spm) = clamp(134 + 2.9 · velocità(km/h), 150, 190)
BPM_target   = cadenza            (entrainment 1:1; anche ½× e 2×)
```
- ✅ **Van Dyck et al. 2015** (*Sports Medicine – Open*): i corridori sincronizzano
  spontaneamente il passo al beat; cadenza naturale ~150–200 spm.
- ✅ **Van Dyck & Moens et al. 2018** (*PLOS ONE*): l'entrainment tiene entro **±2.5–3%**.
- ⚙️ coefficienti `134 / 2.9` = regressione calibrata; clamp 150–190 dai range naturali.

### N3. Doppio regime (invariato dal design originale)
Con velocità/passo dichiarati → `target_bpm` "chirurgico" (N2, quantitativo); senza numeri →
banda BPM del tipo in `GOAL_PARAMS` (qualitativo). `GOAL_PARAMS` (banda, energia, pesi, τ) = ⚙️ design.

## Stadio 2 — Sensori: sforzo e trend (`physiological_state.py`)

### S1. Sforzo dal cuore (HRR) e zone   `compute_hrr / classify_effort`
```
HRR = (HR − HR_rest) / (HR_max − HR_rest)                 # Karvonen
zone:  HRR<0.40 Low · <0.70 Target · <0.85 High · ≥0.85 VeryHigh
```
- ✅ **Karvonen, Kentala & Mustala 1957** — Heart Rate Reserve.
- ✅ **Tanaka et al. 2001** — HR_max = 208 − 0.7·età; **Fox et al. 1971** — 220 − età.
- ⚙️ soglie 0.40/0.70/0.85 = zone d'allenamento standard (design). Le stesse 4 classi sono il
  vocabolario `matches_effort` di `songs.csv`: sensori e catalogo parlano la stessa lingua.

### S2. Trend del cuore (regressione lineare)   `calculate_linear_slope / classify_trend`
```
pendenza = regressione lineare dei BPM sul tempo
> +0.05 bpm/s → Increasing · < −0.05 → Decreasing · altrimenti Stable
```
- ⚙️ soglia 0.05 bpm/s (~3 bpm) = design.

## Mood → generi: dizionario Python (`genre_mood.py`)

### O1. Associazione mood → generi (per la modalità qualitativa) — knowledge-driven
Era un file OWL letto a regex: funzionalmente era già un dizionario (nessuna query, nessuna
inferenza), solo scritto in un formato più complicato — dichiarato per quello che è.
La **regola** resta definita **a priori dalla teoria**, non dedotta dai dati:
- ✅ **Russell 1980** — piano valenza × arousal: ogni mood è una **regione** di quel piano.
- ✅ **Karageorghis & Terry 2009** — l'arousal musicale dipende da tempo/energia: ogni
  **famiglia di generi** ha un archetipo (arousal, valenza) noto dalla letteratura.
- **Regola**: `genres_for_mood(mood)` = i generi il cui archetipo cade nella regione di Russell
  di quel mood (es. metal/punk → alta attivazione → Energetic; ambient/classical → bassa
  attivazione → Calm; pop/latin → valenza alta → Motivated).
- ✅ **Rada et al. 1989** — i generi come rete di concetti (idea ripresa nella regola per famiglie).

`songs.csv` fornisce solo l'**elenco dei generi esistenti** (non le associazioni mood, che
vengono dalla regola teorica sopra): il dizionario può poi essere **validato** contro
`supports_mood` (test set), non è costruito da esso. ⚙️ gli archetipi per famiglia = design → ablation.

Uso: in regime QUALITATIVO (nessun BPM 'chirurgico'), `genres_for_mood(mood)` restituisce i
generi candidati fra cui il recommender pesca le canzoni.

## Controller: generazione del vettore target (`controller.py`)

### C1. Fusione testo ↔ sensori e regimi
`decide(intent, analysis)` produce il vettore target `[bpm, energy, valence]` + raggio + `tau`.
- **Quantitativo** (velocità *dichiarata nel prompt*): target stretto attorno al `bpm` da
  cadenza (Van Dyck 2015), fisso al valore dichiarato, `tau` basso → **exploit**.
- **Qualitativo** (nessun numero nel prompt): il centro del target **insegue la velocità
  reale del momento** — `live_entrainment_bpm()` usa la **cadenza misurata** (1:1 coi passi,
  Van Dyck continuo) o, in mancanza, la stima dalla velocità (`bpm_from_speed`). Range largo,
  generi ristretti al mood, `tau` più alto → **explore**. Prima era il centro-banda fisso del
  tipo; ora se "parti piano e poi spingi", la musica accelera con te. È un **unico punto di
  verità** (`live_entrainment_bpm`), così la logica non è duplicata. ⚙️ design.
- ✅ **Sutton & Barto** — `tau` è la temperatura del softmax/Boltzmann che regola
  exploration/exploitation: alto = varia, basso = preciso.
- ✅ **Russell 1980** — `valence` target dal mood (`VALENCE_BY_MOOD`).

### C2. Riscaldamento (warm-up)
Nei primi `WARMUP_MIN` (=5) minuti il target sale linearmente da una banda bassa (~120 bpm,
energia 0.20) fino al target previsto dell'allenamento: si parte piano e si accelera prima di
entrare nel lavoro. ⚙️ durata e rampa = design → ablation (buona pratica di allenamento).

### C3. Safety override — strato simbolico (`symbolic.py`, `ontology/runner_state.owl`)
`mean_hrr ≥ soglia` → vettore di **recupero** (bpm → banda facile, energy ≤ 0.30): il cuore
vince sull'intento. La soglia (0.90) NON è una costante Python: è un **dato dichiarato
nell'ontologia** (`ar:CriticalState ar:hasThreshold 0.90`); `symbolic.is_critical_state()`
inietta la HRR osservata come tripla RDF e una **query SPARQL** confronta i due valori dentro
il grafo — è la query a classificare lo stato, non un `if` scritto a mano in `controller.py`.
Nessun reasoner DL (niente dipendenza Java): solo `rdflib`, deterministico e verificabile.
⚙️ soglia 0.90 ancorata al limite fisiologico (design → ablation).

### C4. Constraint Gate SHACL sull'NLP (`symbolic.validate_speed`, `ontology/nlp_shapes.ttl`)
Un numero estratto dal testo può essere fisiologicamente assurdo (refuso, allucinazione:
"corro a 300 km/h"). Invece di un controllo Python, una **SHACL shape** dichiara il vincolo
di dominio (`0 < speed_kmh ≤ 45`, oltre il record mondiale di sprint) e `pyshacl` lo valida:
se violato, il valore viene **scartato esplicitamente** invece di essere silenziosamente
clampato più a valle da `bpm_from_speed`. Pattern identico al "Constraint Gate" (domain/range)
discusso a lezione: l'ontologia non genera dati, li **restringe e valida**.

### C5. Variazione per tipo
IntenseRun/ripetute: alterna veloce/lento rispetto alla canzone precedente (`last_bpm`) →
variabilità di ritmo tipica del lavoro a intervalli. ⚙️ design, resta calcolo numerico in Python
(non simbolico): interpolazioni e clamp non sono adatti a OWL/SWRL.

### C6. Effort gate — Generator→Validator sul Top-K (`symbolic.is_effort_compatible`)
Il recommender (§ sotto) ottimizza la distanza vettoriale `[bpm,energy,valence]` ma **ignora**
la compatibilità di sforzo (`matches_effort` della canzone vs `effort_band` del target): misurato
su un campione ampio di combinazioni goal×mood×sforzo×regime, il Top-1 viola l'`effort_band` nel
**16.7% dei casi (20/120)**. `session.pick_song` filtra le **Top-K** del recommender con
`is_effort_compatible` (query SPARQL, stesso meccanismo di C3), tenendo solo le compatibili
(se nessuna lo è, ripiega su tutte le Top-K e logga la violazione).
- Pattern **Generator→Validator**: il modello statistico (recommender) **sovra-genera** candidati,
  un layer simbolico separato **decide VALID/INVALID** — il recommender non cambia, resta
  distanza+softmax puro.
- Non è una scelta stilistica: la soglia di correzione (16.7%) è misurata prima di implementare
  il gate, non assunta.

### C7. Campionamento reale sul Top-K filtrato (`session.pick_song`)
Il recommender calcola `probability` con un softmax (τ) ma poi **ordina e prende il primo**
elemento — è quindi deterministico: target simili producono **sempre le stesse canzoni**
(misurato: sessioni diverse con target vicini restituivano playlist identiche). `pick_song`
completa il softmax **campionando** (non scegliendo sempre il massimo) fra le candidate
compatibili filtrate da C6, pesando per `probability` — è l'algoritmo di Boltzmann che τ
presuppone (Sutton & Barto), applicato dove il recommender lo calcola ma non lo usa. Il
campionamento è seminato per sessione (`zlib.crc32` del prompt/id sessione): riproducibile
entro la stessa sessione, diverso tra sessioni diverse.

Prima del campionamento, `pick_song` **deduplica** le Top-K per (titolo, artista):
`songs.csv` contiene la stessa traccia con `song_id` diversi (fino a ~45 copie per alcuni
brani) — senza deduplica, l'esclusione per `song_id` da sola lascia rientrare "la stessa"
canzone sotto un id diverso. Quando una canzone è scelta, tutte le sue varianti vengono
escluse dai turni successivi (`load_song_variants`), non solo quel `song_id`.
⚙️ soglia di campionamento (`SAMPLE_POOL`, quante candidate vicine considerare) = design.

## Stadio 3 — Recommender (`recommender.py`)
Riceve il `Target`, calcola la **distanza euclidea pesata** target↔canzone e assegna una
probabilità con un **softmax**:
```
d(s) = √[ w_bpm·((bpm_s−bpm*)/tol)² + w_energy·(energy_s−energy*)² + w_valence·(valence_s−valence*)² ]
P(s) = softmax(−d(s)/τ) = exp(−d(s)/τ) / Σ_j exp(−d(j)/τ)
```
- i pesi `w` sono normalizzati a somma 1; il **BPM è normalizzato da `bpm_tolerance`** (tol
  piccola → scarto di BPM più severo; grande → più permissivo);
- restituisce le **Top-K** canzoni per probabilità, escludendo le tracce recenti (memoria) e,
  in regime qualitativo, filtrando sui **generi del mood** (`genre_mood.py`); la compatibilità di
  sforzo sul Top-K è verificata a valle dall'effort gate (C6), non dal recommender stesso.
- ✅ **Sutton & Barto** — softmax/Boltzmann per exploration/exploitation (τ alto esplora, τ→0 sfrutta).
- ⚙️ distanza euclidea pesata e normalizzazione del BPM via tolleranza = scelte di design → ablation.
- ⚠️ **Limite misurato**: il modulo calcola `P(s)` ma poi **ordina e prende il massimo** — a
  parità di target restituisce sempre la stessa canzone; τ non influisce sulla scelta finale, solo
  sul valore di `probability` riportato. L'exploration/exploitation reale (il campionamento da
  `P(s)`) è implementata a valle, in `session.pick_song` (C7).

---

## References (IEEE)

[1] E. Van Dyck et al., "Spontaneous entrainment of running cadence to music tempo," *Sports Medicine – Open*, 1:15, 2015.
[2] B. Moens, E. Van Dyck et al., "Optimizing beat-synchronized running to music," *PLOS ONE*, 13(12):e0208702, 2018.
[3] M. J. Karvonen, E. Kentala, O. Mustala, "The effects of training on heart rate," *Ann. Med. Exp. Biol. Fenn.*, 35(3):307–315, 1957.
[4] H. Tanaka, K. D. Monahan, D. R. Seals, "Age-predicted maximal heart rate revisited," *J. Am. Coll. Cardiol.*, 37(1):153–156, 2001.
[5] S. M. Fox, J. P. Naughton, W. L. Haskell, "Physical activity and the prevention of coronary heart disease," *Ann. Clin. Res.*, 3:404–432, 1971.
[6] J. A. Russell, "A circumplex model of affect," *J. Personality and Social Psychology*, 39(6):1161–1178, 1980.
[7] C. I. Karageorghis, P. C. Terry, "The psychological, psychophysical and ergogenic effects of music in sport," in *Sport and Exercise Psychology*, 2009.
[8] R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018.
[9] R. Rada, H. Mili, E. Bicknell, M. Blettner, "Development and application of a metric on semantic nets," *IEEE Trans. Systems, Man, and Cybernetics*, 19(1):17–30, 1989.

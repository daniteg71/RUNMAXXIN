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

## Ontologia mood → generi (OWL: `ontology/genre_mood.owl`, `genre_mood.py`)

### O1. Associazione mood → generi (per la modalità qualitativa) — **knowledge-driven**
L'ontologia è uno **strato simbolico** (OWL: `owl:Class` Mood/Genre, `owl:ObjectProperty`
genreSuitsMood/dominantMood, `owl:NamedIndividual`). Le associazioni mood↔genere sono definite
**a priori dalla teoria**, non dedotte dai dati:
- ✅ **Russell 1980** — piano valenza × arousal: ogni mood è una **regione** di quel piano.
- ✅ **Karageorghis & Terry 2009** — l'arousal musicale dipende da tempo/energia: ogni
  **famiglia di generi** ha un archetipo (arousal, valenza) noto dalla letteratura.
- **Regola**: `genreSuitsMood(genre)` = i mood la cui regione di Russell contiene l'archetipo
  del genere (es. metal/punk → alta attivazione → Energetic; ambient/classical → bassa
  attivazione → Calm; pop/latin → valenza alta → Motivated).
- ✅ **Rada et al. 1989** — l'ontologia dei generi come rete di concetti.

Separazione **T-Box / A-Box** (Lezoche): la **T-Box** (classi, proprietà, regole di affinità) è
knowledge-driven, definita a priori; il **catalogo** `songs.csv` popola solo le **istanze**
(quali generi esistono, A-Box) e può poi **validare** l'ontologia (accordo con `supports_mood`
come test set), NON la costruisce. ⚙️ gli archetipi per famiglia = scelta di design → ablation.

Uso: in regime QUALITATIVO (nessun BPM 'chirurgico'), `genres_for_mood(mood)` restituisce i
generi candidati fra cui il recommender pesca le canzoni.

## Controller: generazione del vettore target (`controller.py`)

### C1. Fusione testo ↔ sensori e regimi
`decide(intent, analysis)` produce il vettore target `[bpm, energy, valence]` + raggio + `tau`.
- **Quantitativo** (velocità dichiarata): target stretto attorno al `bpm` da cadenza
  (Van Dyck 2015), `tau` basso → **exploit**.
- **Qualitativo**: range largo, generi ristretti al mood (ontologia), `tau` più alto → **explore**.
- ✅ **Sutton & Barto** — `tau` è la temperatura del softmax/Boltzmann che regola
  exploration/exploitation: alto = varia, basso = preciso.
- ✅ **Russell 1980** — `valence` target dal mood (`VALENCE_BY_MOOD`).

### C2. Riscaldamento (warm-up)
Nei primi `WARMUP_MIN` (=5) minuti il target sale linearmente da una banda bassa (~120 bpm,
energia 0.20) fino al target previsto dell'allenamento: si parte piano e si accelera prima di
entrare nel lavoro. ⚙️ durata e rampa = design → ablation (buona pratica di allenamento).

### C3. Safety override e variazione per tipo
- `mean_hrr ≥ 0.90` → vettore di **recupero** (bpm → banda facile, energy ≤ 0.30): il cuore
  vince sull'intento. ⚙️ soglia 0.90 ancorata al limite fisiologico (design → ablation).
- IntenseRun/ripetute: alterna veloce/lento rispetto alla canzone precedente (`last_bpm`) →
  variabilità di ritmo tipica del lavoro a intervalli. ⚙️ design.

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
  in regime qualitativo, filtrando sui **generi del mood** forniti dall'ontologia.
- ✅ **Sutton & Barto** — softmax/Boltzmann per exploration/exploitation (τ alto esplora, τ→0 sfrutta).
- ⚙️ distanza euclidea pesata e normalizzazione del BPM via tolleranza = scelte di design → ablation.
- ✅ **Rada et al. 1989** — l'ontologia dei generi come rete di concetti (filtro del regime qualitativo).

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

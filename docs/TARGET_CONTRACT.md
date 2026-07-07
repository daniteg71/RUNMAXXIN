# Contratto del Target (controller → recommender)

Per chi costruisce il **recommendation system**. Il controller (`controller.py`) espone
`decide(intent, analysis=None, last_bpm=None, elapsed_min=None) -> Target`. Tu ricevi il
`Target`, calcoli una distanza dai vettori-canzone e assegni probabilità con un softmax.

## 1. Cosa ti esce: `Target`

| Campo | Tipo | Significato | Range |
|---|---|---|---|
| `bpm` | float | BPM desiderato | ~80–200 |
| `energy` | float | energia target | 0.0–1.0 |
| `valence` | float | positività target | 0.0–1.0 |
| `weights` | dict | peso di ogni dimensione nella distanza | `{"bpm","energy","valence"}`, somma ~1 |
| `bpm_tolerance` | float | raggio: quanto "conta" uno scarto di BPM | stretto 5 / largo ~10 |
| `genres` | list[str] | generi ammessi (**vuoto = nessun filtro**) | ⊆ generi di `songs.csv` |
| `tau` | float | temperatura del softmax (explore/exploit) | ~0.1–1.0 |
| `mood` | str | Neutral/Focused/Energetic/Motivated/Calm | — |
| `goal` | str | EasyRun/ModerateRun/IntenseRun | — |
| `effort_band` | tuple[str] | classi `matches_effort` ammesse | ⊆ {Low,Target,High,VeryHigh}Effort |
| `recovery` | bool | safety attivo (cuore alto) | — |
| `regime` | str | quantitative / qualitative / warmup / recovery | — |

`target.as_vector()` → `[bpm, energy, valence]` = il **punto** nello spazio delle canzoni.

## 2. Il vettore-canzone (dal catalogo `songs.csv`)
Le stesse 3 dimensioni sono colonne del CSV:
```
v(song) = [ float(song["bpm"]), float(song["energy"]), float(song["valence"]) ]
```
Categoriche utili: `genre` (filtro `genres`), `matches_effort` (filtro `effort_band`),
`supports_mood`/`supports_goal` (filtri extra opzionali).

## 3. Come consumarlo — UNA formula per tutte le varianti
Non serve codice per-variante: recovery e warmup sono **già dentro il vettore**. Ti bastano i campi.
```
d(song) =  w_bpm     · |bpm_s − t.bpm| / t.bpm_tolerance      # BPM normalizzato dal raggio
         + w_energy  · |energy_s − t.energy|
         + w_valence · |valence_s − t.valence|
         (+ penalità se t.genres ≠ [] e song.genre ∉ t.genres)
         (+ opz. scarta se song.matches_effort ∩ t.effort_band = ∅)

P(song) = softmax( − d(song) / t.tau )      # τ alto = varia, τ basso = preciso
scegli la prossima canzone ∝ P(song), escludendo le già suonate
```
- Il **`bpm_tolerance`** è il normalizzatore del BPM: stretto (5) penalizza forte gli scarti
  (regime quantitativo, "resta preciso"); largo (10) è più permissivo (qualitativo, "varia").
- Opzionale ma consigliato: **correzione d'ottava** sul BPM — `min` su ½×,1×,2× di `bpm_s`
  (una canzone a 90 bpm serve un target a 180 al doppio).

## 4. Le varianti (output reali di `decide()`)

**QUANTITATIVO** — velocità dichiarata: BPM chirurgico, raggio stretto, niente filtro genere, τ basso.
```json
{"bpm":169.0,"energy":0.9,"valence":0.75,"weights":{"bpm":0.8,"energy":0.15,"valence":0.05},
 "bpm_tolerance":5.0,"genres":[],"tau":0.2,"mood":"Energetic","goal":"IntenseRun",
 "effort_band":["HighEffort","VeryHighEffort"],"recovery":false,"regime":"quantitative"}
```

**QUALITATIVO** — nessun numero: raggio largo, **generi del mood**, τ più alto (varia).
```json
{"bpm":175.0,"energy":0.9,"valence":0.75,"weights":{"bpm":0.85,"energy":0.09,"valence":0.06},
 "bpm_tolerance":10.0,"genres":["black-metal","breakbeat","edm","techno","hardstyle","happy", "..."],
 "tau":0.15,"mood":"Energetic","goal":"IntenseRun","effort_band":["HighEffort","VeryHighEffort"],
 "recovery":false,"regime":"qualitative"}
```

**WARMUP** — primi ~5 min: stesso schema, ma `bpm`/`energy` più bassi (rampa). Trattalo come un target normale.
```json
{"bpm":139.6,"energy":0.48,"valence":0.75,"weights":{"bpm":0.8,"energy":0.15,"valence":0.05},
 "bpm_tolerance":5.0,"genres":[],"tau":0.2,"mood":"Energetic","goal":"IntenseRun",
 "effort_band":["HighEffort","VeryHighEffort"],"recovery":false,"regime":"warmup"}
```

**RECOVERY** — safety (cuore alto): `recovery=true`, BPM/energy/valence bassi, `effort_band=(LowEffort,)`.
```json
{"bpm":120.0,"energy":0.3,"valence":0.25,"weights":{"bpm":0.8,"energy":0.15,"valence":0.05},
 "bpm_tolerance":5.0,"genres":[],"tau":0.2,"mood":"Energetic","goal":"IntenseRun",
 "effort_band":["LowEffort"],"recovery":true,"regime":"recovery"}
```

**FUSIONE** (es. EasyRun ma sforzo alto → calma): il vettore è già "abbassato", `weights` dal tipo.
```json
{"bpm":123.8,"energy":0.175,"valence":0.25,"weights":{"bpm":0.2,"energy":0.48,"valence":0.32},
 "bpm_tolerance":7.5,"genres":["ambient","classical","piano","sleep","jazz","..."],
 "tau":1.0,"mood":"Calm","goal":"EasyRun","effort_band":["LowEffort","TargetEffort"],
 "recovery":false,"regime":"qualitative"}
```

## 5. In una frase
Ti arriva **sempre** lo stesso oggetto `Target`; le "varianti" cambiano solo i **valori** di
`bpm/energy/valence`, `bpm_tolerance`, `genres`, `tau`, `effort_band`. La tua funzione è una sola:
distanza pesata su `as_vector()` (BPM normalizzato da `bpm_tolerance`) + filtro `genres`/`effort_band`
→ `softmax(−d/tau)` → prossima canzone.

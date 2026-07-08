# Related Work — research notes (working scaffold, not final prose)

Fonti verificate via ricerca web (luglio 2026). Organizzate secondo i 5 punti richiesti dal
template. Ogni lavoro è collegato al NOSTRO progetto (come ci differenziamo), non elencato a vuoto.

Stato attuale del nostro sistema (da tenere fisso per non ricadere in descrizioni vecchie):
- La RACCOMANDAZIONE è statistica (distanza vettoriale + softmax, modulo del collega).
- L'ONTOLOGIA non raccomanda: è un BORDO deterministico in 3 punti — safety (SPARQL),
  validazione input NLP (SHACL), effort-gate sul Top-K (SPARQL).
- Il mood→genere è un dizionario Python ESPLICITO (fuori dall'ontologia), non inferenza OWL.
- NLP = SetFit few-shot. Sessioni fisiologiche = SIMULATE.

---

## 1. Approcci classici / early (sport + fisiologia + musica)

- **MPTrain** — Oliver & Flores-Mangas, *MobileHCI 2006*. HR + passo → seleziona musica per
  guidare l'utente verso una curva di HR desiderata. Obiettivo = curva numerica, NON linguaggio.
  → Noi: aggiungiamo canale NLP + safety separata/ispezionabile (non fusa nel modello).
- **PAPA** — Oliver & Kreger-Stickles, *ISMIR 2006*. Generalizza MPTrain: playlist "physiology +
  purpose aware". Il "purpose" è predefinito.
  → Noi: il purpose è testo libero classificato few-shot.
- **Affective Music Player** — Janssen, van den Broek & Westerink, *UMUAI 22(3), 2012*. Biosegnali
  → modella la reazione emotiva → sceglie musica verso uno stato d'umore target (regressione+KDE).
  → Noi: è mood-enhancement a riposo, non sforzo atletico real-time con vincolo di sicurezza.

Base teorica (sostiene le nostre scelte, non è "sistema"):
- **Karageorghis, Terry, Lane et al.** — *"Music in the exercise domain: a review and synthesis
  (Part I & II)", Int. Rev. Sport Exerc. Psychol., 2012*. Framework di riferimento. Dato chiave:
  banda motivazionale 125–140 bpm per 40–90% HRR (uso asincrono); l'uso SINCRONO (movimento
  coordinato al beat) ha effetti ergogenici superiori. → giustifica il nostro entrainment 1:1
  cadenza↔beat (Van Dyck) invece del semplice tempo-matching.

## 2. Metodi deep / AI recenti

- **Deep content-based music recommendation** — van den Oord, Dieleman & Schrauwen, *NIPS 2013*.
  CNN su spettrogrammi → latent factors, risolve il cold-start. Seminale (1000+ cit.).
  → Noi: la SOTA va verso modelli audio pesanti/opachi; noi andiamo verso leggero+ispezionabile,
    perché il nostro collo di bottiglia è la sicurezza real-time, non l'accuratezza di preferenza.
- **SetFit** — Tunstall, Reimers, Jo, Bates, Korat, Wasserblat, Pereg, *2022 (arXiv 2209.11055)*.
  Few-shot senza prompt: fine-tuning contrastivo di un Sentence-Transformer + testa leggera.
  Accuratezza alta con ordini di grandezza meno parametri, poche decine di esempi/classe.
  → È LA NOSTRA architettura NLP. Citazione obbligatoria.
  → FIGURE ufficiali (da attribuire "adapted from Tunstall et al."): (a) schema architettura,
    (b) grafico CR accuracy vs #esempi/classe, SetFit_MPNET >> RoBERTa_large in low-data.
- **LLM per sensor fusion** — Apple ML 2025; *SensorLLM (arXiv 2410.10624)*. LLM che fondono serie
  temporali da sensori per activity recognition.
  → Noi: "NLP + dati sensori" è tema caldo/aperto, ma loro usano LLM giganti; noi fusione leggera
    e mirata (testo per l'intento, sensori numerici per lo stato — non un LLM sui segnali).

## 3. Ontologie / knowledge-based (confronto CRITICO, il più vicino sul lato simbolico)

- **COMUS** — Song, Kim et al., *"COMUS: Ontological and Rule-Based Reasoning for Music
  Recommendation System"* + *"Music Ontology for Mood and Situation Reasoning"* (IEEE). Ontologia
  OWL che estende la Music Ontology con mood+situazione; usa OWL+SPARQL+REGOLE per raccomandare
  ragionando su umore/contesto. Colma il semantic gap feature-basse↔emozioni-alte.
  → DIFFERENZA CHIAVE (stato attuale, NON quello vecchio): in COMUS l'ontologia È il motore di
    raccomandazione (inferisce mood+situazione→musica con un reasoner a regole). Da noi l'ontologia
    NON raccomanda: la raccomandazione resta statistica, e il simbolico è solo un BORDO
    deterministico (safety, validazione input, effort-gate). La conoscenza mood→genere che COMUS
    mette DENTRO il grafo, noi la teniamo esplicita FUORI (dizionario), riservando il livello
    simbolico ai vincoli da GARANTIRE, non da inferire. Inoltre COMUS è per ascolto generico, senza
    vincolo fisiologico di sicurezza né funzionamento durante lo sforzo.
- **Neuro-symbolic pipeline** — Kutt et al., *"A Three-stage Neuro-symbolic Recommendation Pipeline
  for Cultural Heritage KG", arXiv 2026*. SPARQL + KG + componente neurale.
  → Stesso PATTERN nostro (query simbolica + statistica), dominio diverso (beni culturali statici,
    non sicurezza fisiologica real-time nello sport).

## 4. Baseline pratiche (repository GitHub — cosa fa chi implementa senza ricerca)

- **heartBPMusic** (ColinWu0403): Django+Vue+scikit-learn k-NN. BPM manuale, no NLP, NO SAFETY.
- **HeartBEAT** (mray190): mbed+BLE+Android. Matching diretto BPM↔battito, NO override sicurezza.
- **Tempo-aware Music Rec** (coursework): RNN+Fitbit API. No intento, no safety.
  → Verificati aprendo i repo. Dimostrano il vuoto che riempiamo (fusione+NLP+safety).

## 5. Dataset, benchmark, protocolli di valutazione

Per la NOSTRA purpose ci sono DUE parti valutabili → due famiglie di benchmark/protocolli:

NLP (classificazione d'intento):
- **RAFT** — *"A Real-World Few-Shot Text Classification Benchmark", NeurIPS 2021 (arXiv
  2109.14076)*. Protocollo few-shot realistico: ~50 esempi, NIENTE validation set, metrica
  MACRO-F1 (per gestire sbilanciamento di classe).
  → È il protocollo che il nostro eval segue concettualmente (test tenuto fuori, macro-F1,
    baseline a regole). Lo citiamo come standard di valutazione few-shot.

Recommendation:
- **Million Song Dataset** — Bertin-Mahieux, Ellis, Whitman, Lamere, *ISMIR 2011*. ~1M tracce,
  il benchmark di comunità per music rec.
  → ONESTÀ: NON lo usiamo. Il nostro catalogo è un export di feature Spotify (songs.csv, ~89.5k
    tracce con bpm/energy/valence). Lo citiamo come benchmark di riferimento del campo, dichiarando
    che il nostro contesto (fusione fisiologica + safety) non ha un dataset standard.
- **Metriche beyond-accuracy** (diversity, novelty, coverage) — standard per valutare rec oltre
  l'accuratezza (survey evaluation techniques, arXiv 2312.16015).
  → Giustificano il nostro fix dedup+sampling: aumenta diversità/novelty della playlist (prima il
    recommender deterministico ripeteva le stesse canzoni).
- **SVR@K — Safety Violation Ratio** — *SafeCRS (arXiv 2603.03536)*: percentuale di raccomandazioni
  che violano un vincolo di sicurezza dell'utente (+ S-DCG@K che pesa di più le violazioni in alto).
  → LEGITTIMA il nostro 16.7%: il nostro effort-gate misura esattamente un Safety Violation Ratio.
    Non è una metrica inventata da noi, è un tipo riconosciuto in letteratura.
- **Health/safety-aware recommenders** — review (Springer AI Review 2026): nella valutazione dei
  sistemi di raccomandazione sanitari la "safety" pesa più di ogni altro indicatore (0.289).
  → Sostiene la nostra tesi: in dominio a rischio, la sicurezza è un vincolo primario, non un extra.

## Limitazioni REALI della letteratura (per il punto 4 + Critical Discussion)

1. Nessun DATASET STANDARD per music-rec fisiologico: campioni piccoli (<100 partecipanti tipici),
   spesso su musica classica occidentale; i testi sono coperti da copyright → riproducibilità
   difficile (dataset non condivisibili). [survey emotional music DB; LLM music rec challenges 2025]
   → Noi condividiamo lo stesso limite (sessioni SIMULATE), ma lo dichiariamo apertamente.
2. I sistemi classici (MPTrain/PAPA) NON hanno canale linguistico: l'obiettivo va impostato a numeri.
3. I sistemi ontologici (COMUS) mettono TUTTO nel reasoner → pesante, non pensato per aggiornamenti
   continui ad alta frequenza né per la sicurezza real-time.
4. Le implementazioni pratiche (repo GitHub) non hanno NÉ valutazione NÉ safety.
5. La valutazione offline dei rec ignora spesso i vincoli di sicurezza/contesto (health-rec review):
   la maggior parte ottimizza solo accuratezza/ranking.

## Come la NOSTRA proposta differisce (sintesi punto 5)

- Aggiunge un canale d'intento in LINGUAGGIO NATURALE assente in tutti i sistemi rivisti.
- Usa il livello simbolico come BORDO di sicurezza/validazione (non come motore), scelta opposta a
  COMUS e più adatta al real-time.
- MISURA (non assume) una violazione reale del recommender statistico (16.7% ≈ un SVR) e la corregge
  con un layer di ricampionamento leggero → migliora anche diversity/novelty.
- Valuta l'NLP con protocollo few-shot (macro-F1 + baseline a regole), nello spirito di RAFT.

## Citazioni: nucleo forte vs supporto

Nucleo (asse del Related Work):
  MPTrain/PAPA (2006) · COMUS (Song et al.) · SetFit (Tunstall 2022) · van den Oord (2013) ·
  Karageorghis (2012) · Kutt (2026).
Supporto:
  Million Song Dataset (2011) · RAFT (2021) · SVR@K/SafeCRS (2026) · Janssen (2012) ·
  beyond-accuracy survey · repo GitHub (heartBPMusic, HeartBEAT).

## Figure SetFit (da collocare)
- Architettura SetFit → Methodology (modulo NLP) oppure Related Work. Attribuire a Tunstall et al.
- Grafico CR (SetFit vs RoBERTa-large, low-data) → Related Work, a supporto di "few-shot invece di
  fine-tuning di un Transformer grande". Attribuire a Tunstall et al.

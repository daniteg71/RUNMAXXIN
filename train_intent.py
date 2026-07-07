"""Allena i due classificatori d'intento di RUNMAXXIN (SetFit few-shot, gira su M2 in minuti).

Due modelli, entrambi ancorati al vocabolario di `songs.csv`:
  - GOAL  (3 classi): EasyRun / ModerateRun / IntenseRun     -> tipo di allenamento
  - MOOD  (5 classi): Neutral/Focused/Energetic/Motivated/Calm

Stessa impostazione del trainer di AlgoRun (SetFit + encoder multilingue italiano); qui
cambiano solo le etichette (riaddestrate sul CSV) e si allena anche il mood.

Uso:  python train_intent.py    # salva in models/intent-goal-setfit e models/intent-mood-setfit
"""
from pathlib import Path

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

BASE = Path(__file__).parent
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# GOAL: frase -> tipo di allenamento (supports_goal). ~12 frasi/classe.
# ---------------------------------------------------------------------------
GOAL_TRAIN = [
    ("oggi corsa piano di recupero", "EasyRun"), ("qualcosa di tranquillo, sono scarico", "EasyRun"),
    ("corsetta chill senza fatica", "EasyRun"), ("voglio solo sciogliere le gambe", "EasyRun"),
    ("corsa lenta e rilassata", "EasyRun"), ("oggi vado piano piano", "EasyRun"),
    ("recupero blando, niente sforzo", "EasyRun"), ("corsa leggera per rigenerarmi", "EasyRun"),
    ("giro tranquillo di venti minuti", "EasyRun"), ("oggi relax, corsa facile", "EasyRun"),
    ("defaticante, molto piano", "EasyRun"), ("una sgambata leggera senza sudare", "EasyRun"),

    ("oggi faccio il lungo a ritmo costante", "ModerateRun"), ("corsa media, spingo un po'", "ModerateRun"),
    ("fondo lento di un'ora", "ModerateRun"), ("ritmo medio deciso ma gestibile", "ModerateRun"),
    ("tanti chilometri a passo regolare", "ModerateRun"), ("tempo run a ritmo sostenuto", "ModerateRun"),
    ("corsa costante per la resistenza", "ModerateRun"), ("endurance a ritmo tenuto", "ModerateRun"),
    ("medio veloce per quaranta minuti", "ModerateRun"), ("corsa impegnativa ma costante", "ModerateRun"),
    ("dieci chilometri a ritmo gara controllato", "ModerateRun"), ("passo regolare e continuo per un'ora", "ModerateRun"),

    ("oggi ripetute veloci", "IntenseRun"), ("intervalli e scatti al massimo", "IntenseRun"),
    ("sei volte i mille metri", "IntenseRun"), ("scatti forti e recupero", "IntenseRun"),
    ("ripetute brevi ad alta intensità", "IntenseRun"), ("allenamento a intervalli esplosivi", "IntenseRun"),
    ("sprint e pausa, ripetuti", "IntenseRun"), ("oggi si scatta a tutta", "IntenseRun"),
    ("fartlek con variazioni e accelerazioni forti", "IntenseRun"), ("un minuto a palla e uno piano", "IntenseRun"),
    ("massimo sforzo a ripetizioni", "IntenseRun"), ("quattrocento metri veloci con recupero", "IntenseRun"),
]

GOAL_TEST = [
    ("vado piano oggi, solo recupero", "EasyRun"), ("corsetta blanda per sciogliermi", "EasyRun"),
    ("macino chilometri a ritmo costante", "ModerateRun"), ("medio deciso per mezz'ora", "ModerateRun"),
    ("ripetute da quattrocento veloci", "IntenseRun"), ("scatti al massimo con pause", "IntenseRun"),
    ("Andiamo a smaltire la pizza senza fretta", "EasyRun"),
    ("Fammela sudare tanto ma a ritmo costante e regolare", "ModerateRun"),
    ("Voglio il cuore in gola a fasi alterne con scatti", "IntenseRun"),
]

# ---------------------------------------------------------------------------
# MOOD: frase -> stato d'animo (supports_mood). ~11 frasi/classe.
# ---------------------------------------------------------------------------
MOOD_TRAIN = [
    ("oggi una corsa normale, niente di particolare", "Neutral"), ("nessuna emozione speciale, corro e basta", "Neutral"),
    ("una corsa qualsiasi come le altre", "Neutral"), ("stato d'animo neutro, si va", "Neutral"),
    ("non mi sento in un modo particolare oggi", "Neutral"), ("corro senza pensieri, così così", "Neutral"),
    ("giornata qualunque, corsa di routine", "Neutral"), ("nè carico nè stanco, nella media", "Neutral"),
    ("oggi tutto regolare, umore piatto", "Neutral"), ("corsa ordinaria, niente da segnalare", "Neutral"),
    ("mi sento normale, procedo tranquillo", "Neutral"),

    ("sono super concentrato oggi", "Focused"), ("testa bassa e massima concentrazione", "Focused"),
    ("voglio restare in zona tutto il tempo", "Focused"), ("focus totale sull'allenamento", "Focused"),
    ("lucido e concentrato sul ritmo", "Focused"), ("mente ferma, attento a ogni passo", "Focused"),
    ("oggi sono nel flow, concentratissimo", "Focused"), ("solo io e la corsa, zero distrazioni", "Focused"),
    ("determinato a restare concentrato", "Focused"), ("attenzione al cento per cento", "Focused"),
    ("resto focalizzato sull'obiettivo del passo", "Focused"),

    ("oggi sono carico a mille", "Energetic"), ("pieno di energia, scoppio di forza", "Energetic"),
    ("gasato e pronto a spaccare", "Energetic"), ("mi sento elettrico, tanta energia", "Energetic"),
    ("sprizzo energia da tutti i pori", "Energetic"), ("oggi ho una carica pazzesca", "Energetic"),
    ("adrenalina alle stelle, vado", "Energetic"), ("mi sento una molla, super energico", "Energetic"),
    ("batterie cariche, si parte forte", "Energetic"), ("che carica oggi, energia pura", "Energetic"),
    ("mi sento esplosivo e pieno di grinta fisica", "Energetic"),

    ("ho tanta voglia di spingere oggi", "Motivated"), ("determinato a dare tutto me stesso", "Motivated"),
    ("grinta e voglia di migliorarmi", "Motivated"), ("oggi voglio superare i miei limiti", "Motivated"),
    ("motivatissimo, niente mi ferma", "Motivated"), ("voglio battere il mio record", "Motivated"),
    ("carico di determinazione, si spinge", "Motivated"), ("oggi mi impegno al massimo", "Motivated"),
    ("voglia di sudare e crescere", "Motivated"), ("obiettivo chiaro, do il cento per cento", "Motivated"),
    ("pronto a lottare per il risultato", "Motivated"),

    ("oggi sono rilassato e sereno", "Calm"), ("mi sento tranquillo e in pace", "Calm"),
    ("calmo, senza nessuno stress", "Calm"), ("voglio una corsa zen e rilassante", "Calm"),
    ("mente serena, respiro calmo", "Calm"), ("oggi cerco relax e tranquillità", "Calm"),
    ("mi sento disteso e leggero", "Calm"), ("nessuna ansia, tutto placido", "Calm"),
    ("corsa per rilassarmi e staccare", "Calm"), ("umore quieto e disteso", "Calm"),
    ("sono in tranquillità assoluta", "Calm"),
]

MOOD_TEST = [
    ("corsa normale, umore nella media", "Neutral"), ("niente di che, mi sento neutro", "Neutral"),
    ("sono concentratissimo sul passo", "Focused"), ("resto in zona senza distrazioni", "Focused"),
    ("che carica, sono gasato oggi", "Energetic"), ("pieno di energia e adrenalina", "Energetic"),
    ("voglia di spingere e superarmi", "Motivated"), ("determinato a dare tutto", "Motivated"),
    ("tranquillo e rilassato, corsa zen", "Calm"), ("sereno e in pace, senza stress", "Calm"),
]


def _train_one(train, test, out_dir: Path, name: str) -> None:
    ds = Dataset.from_dict({"text": [t for t, _ in train], "label": [l for _, l in train]})
    model = SetFitModel.from_pretrained(ENCODER)
    trainer = Trainer(model=model, args=TrainingArguments(batch_size=16, num_epochs=1),
                      train_dataset=ds)
    trainer.train()
    model.save_pretrained(str(out_dir))

    ok = sum(model.predict([t])[0] == g for t, g in test)
    print(f"\n[{name}] accuracy sul test tenuto fuori: {ok}/{len(test)}")
    for t, g in test:
        p = model.predict([t])[0]
        print(f"  {'OK ' if p == g else 'X  '}{t!r:48} -> {p}  (atteso {g})")


def main() -> None:
    _train_one(GOAL_TRAIN, GOAL_TEST, BASE / "models" / "intent-goal-setfit", "GOAL")
    _train_one(MOOD_TRAIN, MOOD_TEST, BASE / "models" / "intent-mood-setfit", "MOOD")


if __name__ == "__main__":
    main()

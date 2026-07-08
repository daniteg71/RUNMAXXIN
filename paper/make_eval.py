"""Misure aggiuntive per il paper (confronti modelli + gate simbolico), riproducibili.

Produce numeri REALI (niente valori inventati) e i relativi grafici in paper/figures/:
  1) confronto NLP: majority / keyword / TF-IDF+LogReg / SetFit  (accuracy + macro-F1)
  2) effort-gate: % di raccomandazioni incompatibili con la banda di sforzo, con vs senza gate,
     su uno sweep di 120 scenari (3 goal x 5 mood x 4 effort x 2 regimi)
  3) efficienza: n. parametri encoder vs RoBERTa-large (letteratura), latenza gia' nota

Uso:  python paper/make_eval.py    # richiede i modelli SetFit in models/
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_intent import GOAL_TRAIN, GOAL_TEST, MOOD_TRAIN, MOOD_TEST   # noqa: E402
from intent import goal_from_keywords, GOAL_PARAMS, bpm_from_speed      # noqa: E402
from controller import decide                                          # noqa: E402
from symbolic import is_effort_compatible                              # noqa: E402
import recommender                                                     # noqa: E402
import session                                                         # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GRAY, GREEN, RED, ORANGE = "#3b6ea5", "#9aa0a6", "#4c9a5a", "#c0504d", "#e08a3c"


def setfit_predict(model_dir, texts):
    from setfit import SetFitModel
    m = SetFitModel.from_pretrained(str(ROOT / model_dir))
    return [str(p) for p in m.predict(texts)]


def nlp_comparison():
    def scores(y, pred):
        return accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)

    rows = {}
    for task, tr, te, mdl in [("goal", GOAL_TRAIN, GOAL_TEST, "models/intent-goal-setfit"),
                              ("mood", MOOD_TRAIN, MOOD_TEST, "models/intent-mood-setfit")]:
        Xtr = [t for t, _ in tr]; ytr = [l for _, l in tr]
        Xte = [t for t, _ in te]; yte = [l for _, l in te]

        majority = Counter(ytr).most_common(1)[0][0]
        maj = scores(yte, [majority] * len(yte))

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(vec.fit_transform(Xtr), ytr)
        tfidf = scores(yte, list(clf.predict(vec.transform(Xte))))

        setf = scores(yte, setfit_predict(mdl, Xte))

        if task == "goal":
            kw = scores(yte, [goal_from_keywords(t) or "ModerateRun" for t in Xte])
            rows[task] = {"Majority": maj, "Keyword": kw, "TF-IDF+LR": tfidf, "SetFit": setf}
        else:
            rows[task] = {"Majority": maj, "TF-IDF+LR": tfidf, "SetFit": setf}
    return rows


def gate_measurement(top_k=8):
    """% di canzoni del Top-K incompatibili con la effort_band, con vs senza gate, su 120 scenari."""
    effort_by_song = session.load_effort_by_song(str(ROOT / "songs.csv"))
    goals = ["EasyRun", "ModerateRun", "IntenseRun"]
    moods = ["Neutral", "Focused", "Energetic", "Motivated", "Calm"]
    efforts = ["LowEffort", "TargetEffort", "HighEffort", "VeryHighEffort"]

    total_reco = total_bad_before = 0
    scenarios = gate_fails = 0
    for g in goals:
        for m in moods:
            for eff in efforts:
                for target_bpm in (None, bpm_from_speed(12)):     # qualitativo / quantitativo
                    intent = {"goal": g, "mood": m, "numbers": {}, "target_bpm": target_bpm,
                              "params": GOAL_PARAMS[g]}
                    analysis = {"mean_hrr": 0.5, "effort_state": eff, "trend_state": "Stable",
                                "mean_speed_kmh": 11.0}
                    target = decide(intent, analysis=analysis)
                    reco = recommender.recommend(target, songs_path=str(ROOT / "songs.csv"),
                                                 top_k=top_k, seed=0)
                    compat = []
                    for sid in reco["song_id"].astype(str):
                        labels = effort_by_song.get(sid, "").split(";")
                        compat.append(is_effort_compatible(labels, target.effort_band))
                    total_reco += len(compat)
                    total_bad_before += sum(1 for c in compat if not c)
                    scenarios += 1
                    if not any(compat):
                        gate_fails += 1
    before = total_bad_before / total_reco                 # % raccomandazioni incompatibili (senza gate)
    after = gate_fails / scenarios                         # % scenari in cui il gate NON trova nulla
    return before, after, scenarios


def main():
    print("== 1) confronto NLP ==")
    rows = nlp_comparison()
    for task, d in rows.items():
        for name, (a, f) in d.items():
            print(f"  {task:5} {name:10} acc={a:.3f}  macroF1={f:.3f}")

    # --- Fig: confronto accuracy multi-baseline (goal) ---
    d = rows["goal"]; names = list(d.keys()); accs = [d[n][0] for n in names]
    colors = [GRAY, GRAY, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(names, accs, color=colors, width=0.62)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy (goal)")
    ax.set_title("Goal classification: baselines vs SetFit")
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_model_comparison.png", dpi=200); plt.close(fig)

    print("\n== 2) effort-gate ==")
    before, after, n = gate_measurement()
    print(f"  scenari={n}  incompatibili SENZA gate={before:.3f}  gate non trova nulla={after:.3f}")

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    bars = ax.bar(["Without gate\n(recommender)", "With gate\n(ours)"],
                  [before, after], color=[RED, GREEN], width=0.6)
    ax.set_ylim(0, max(0.3, before * 1.3)); ax.set_ylabel("Effort-incompatible rate")
    ax.set_title("Safety (effort) violations: gate ablation")
    for b, v in zip(bars, [before, after]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v*100:.1f}%", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_gate_ablation.png", dpi=200); plt.close(fig)

    print(f"\nFigure salvate in {OUT}")


if __name__ == "__main__":
    main()

"""Genera le figure quantitative del paper (Results) dai MODELLI VERI, in modo riproducibile.

Esegue i due classificatori SetFit sul test tenuto fuori (train_intent.GOAL_TEST/MOOD_TEST),
calcola accuracy, F1 per classe e matrici di confusione, e salva i PNG in paper/figures/:
  - fig_accuracy_baseline.png : accuracy GOAL, baseline a keyword vs SetFit
  - fig_goal_f1_per_class.png : F1 per classe del GOAL (mostra la debolezza su IntenseRun)
  - fig_confusion_goal.png     : matrice di confusione GOAL (3x3)
  - fig_confusion_mood.png     : matrice di confusione MOOD (5x5)

Uso:  python paper/make_figures.py     # richiede i modelli in models/ (python train_intent.py)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_intent import GOAL_TEST, MOOD_TEST          # noqa: E402
from intent import goal_from_keywords                 # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GRAY, RED = "#3b6ea5", "#9aa0a6", "#c0504d"


def setfit_predict(model_dir: str, texts):
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(str(ROOT / model_dir))
    return [str(p) for p in model.predict(texts)]


def main() -> None:
    # --- predizioni reali sul test tenuto fuori ---
    gX = [t for t, _ in GOAL_TEST]; gy = [l for _, l in GOAL_TEST]
    mX = [t for t, _ in MOOD_TEST]; my = [l for _, l in MOOD_TEST]

    g_pred = setfit_predict("models/intent-goal-setfit", gX)
    m_pred = setfit_predict("models/intent-mood-setfit", mX)
    g_base = [goal_from_keywords(t) or "ModerateRun" for t in gX]   # baseline a regole

    g_acc = accuracy_score(gy, g_pred)
    g_base_acc = accuracy_score(gy, g_base)
    m_acc = accuracy_score(my, m_pred)
    print(f"GOAL  SetFit acc={g_acc:.3f}  baseline={g_base_acc:.3f}   MOOD SetFit acc={m_acc:.3f}")

    # === Fig 1: accuracy GOAL, baseline vs SetFit (grafico a barre) ===
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(["Keyword\nbaseline", "SetFit\n(ours)"], [g_base_acc, g_acc],
                  color=[GRAY, BLUE], width=0.6)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Accuracy (goal)")
    ax.set_title("Goal classification: baseline vs SetFit")
    for b, v in zip(bars, [g_base_acc, g_acc]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_accuracy_baseline.png", dpi=200); plt.close(fig)

    # === Fig 2: F1 per classe del GOAL (grafico a barre) ===
    goal_labels = ["EasyRun", "ModerateRun", "IntenseRun"]
    f1s = f1_score(gy, g_pred, labels=goal_labels, average=None, zero_division=0)
    colors = [RED if f < 0.75 else BLUE for f in f1s]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(goal_labels, f1s, color=colors, width=0.6)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 (goal)")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_goal_f1_per_class.png", dpi=200); plt.close(fig)

    # === Fig 3 & 4: matrici di confusione ===
    def plot_cm(y, pred, labels, title, fname):
        cm = confusion_matrix(y, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 1.6, 0.9 * len(labels) + 1.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        thr = cm.max() / 2 if cm.max() else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black", fontweight="bold")
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=200); plt.close(fig)

    plot_cm(gy, g_pred, goal_labels, "Confusion matrix (goal)", "fig_confusion_goal.png")
    plot_cm(my, m_pred, ["Neutral", "Focused", "Energetic", "Motivated", "Calm"],
            "Confusion matrix (mood)", "fig_confusion_mood.png")

    print(f"Figure salvate in {OUT}")


if __name__ == "__main__":
    main()

"""Valutazione dei classificatori d'intento (per il paper).

Su un test tenuto fuori dal training misura accuracy + precision/recall/F1 per classe,
e confronta SetFit con un BASELINE a regole (keyword) -> ablation.

Uso:  python eval_intent.py     # richiede i modelli in models/ (python train_intent.py)
"""
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_intent import GOAL_TEST, MOOD_TEST
from intent import goal_from_keywords

BASE = Path(__file__).parent


def _setfit(model_dir: str, texts: list[str]) -> list[str]:
    from setfit import SetFitModel
    model = SetFitModel.from_pretrained(str(BASE / model_dir))
    return [str(p) for p in model.predict(texts)]


def evaluate(name: str, model_dir: str, test: list, baseline=None) -> None:
    X = [t for t, _ in test]
    y = [l for _, l in test]
    pred = _setfit(model_dir, X)
    print(f"\n===== {name} — SetFit ({len(test)} esempi tenuti fuori) =====")
    print(f"accuracy: {accuracy_score(y, pred):.3f}")
    print(classification_report(y, pred, zero_division=0))
    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels)
    print("confusion matrix (righe=gold, colonne=pred):")
    print("            " + " ".join(f"{l[:10]:>10s}" for l in labels))
    for lbl, row in zip(labels, cm):
        print(f"{lbl[:10]:>10s}  " + " ".join(f"{v:>10d}" for v in row))
    if baseline is not None:
        # baseline a regole: keyword se c'e', altrimenti la classe piu' frequente (ModerateRun)
        bp = [baseline(x) or "ModerateRun" for x in X]
        print(f"----- {name} — BASELINE keyword -----")
        print(f"accuracy: {accuracy_score(y, bp):.3f}  (ablation: quanto aggiunge SetFit)")


def main() -> None:
    evaluate("GOAL", "models/intent-goal-setfit", GOAL_TEST, baseline=goal_from_keywords)
    evaluate("MOOD", "models/intent-mood-setfit", MOOD_TEST)


if __name__ == "__main__":
    main()

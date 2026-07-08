"""Genera figures/pipeline.png: il diagramma dell'architettura RUNMAXXIN nello stile del
template del corso (box numerati, accento blu, barra Experimental Protocol, legenda)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2f5fa6"       # accento
HL = "#e6eefaff"       # riempimento box contributo
GRAYE = "#aeb6c2"      # bordo neutro
DARK = "#1a2330"       # testo

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, header, bullets, highlight=False):
    ec = BLUE if highlight else GRAYE
    fc = HL if highlight else "white"
    lw = 2.2 if highlight else 1.3
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))
    hc = BLUE if highlight else DARK
    ax.text(x + 0.18, y + h - 0.33, header, fontsize=12.5, fontweight="bold", color=hc, va="top")
    ax.plot([x + 0.16, x + w - 0.16], [y + h - 0.62, y + h - 0.62], color=GRAYE, lw=0.8)
    for i, b in enumerate(bullets):
        ax.text(x + 0.22, y + h - 0.95 - i * 0.46, b, fontsize=10.2, color=DARK, va="top")


def arrow(x1, y1, x2, y2, color=BLUE, dashed=False, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=lw, linestyle="--" if dashed else "-"))


# --- titolo ---
ax.text(8, 8.6, "RUNMAXXIN: Architecture and Experimental Pipeline",
        fontsize=17, fontweight="bold", color=DARK, ha="center")

# --- 6 stadi ---
w, h, g, x0, ytop = 2.36, 2.9, 0.26, 0.35, 4.7
xs = [x0 + i * (w + g) for i in range(6)]

box(xs[0], ytop, w, h, "1. Input",
    ["Text prompt", "Heart rate", "Speed / cadence"])
box(xs[1], ytop, w, h, "2. Preprocessing",
    ["Regex numbers", "30 s windows", "Catalogue clean"])
box(xs[2], ytop, w, h, "3. Features",
    ["SetFit goal/mood", "HRR (Karvonen)", "effort / trend"])
box(xs[3], ytop, w, h, "4. Controller", ["Fuse text +", "sensors ->", "[BPM,Ener,Val]"], highlight=True)
box(xs[4], ytop, w, h, "5. Recommender",
    ["Weighted dist.", "softmax sample", "Top-K tracks"])
box(xs[5], ytop, w, h, "6. Safety Gate", ["SPARQL effort", "HRR override", "-> chosen track"], highlight=True)

ymid = ytop + h / 2
for i in range(5):
    arrow(xs[i] + w, ymid, xs[i + 1], ymid)

# --- override di sicurezza (dashed): dai sensori/gate torna al controller ---
arrow(xs[5] + w / 2, ytop, xs[3] + w / 2, ytop, color="#c0504d", dashed=True, lw=1.8)
ax.text((xs[3] + xs[5]) / 2 + w / 2, ytop - 0.42, "safety override (HRR $\\geq$ 0.90)",
        fontsize=9.5, color="#c0504d", ha="center", style="italic")

# --- legenda ---
ax.add_patch(FancyArrowPatch((0.5, 3.15), (1.4, 3.15), arrowstyle="-|>", mutation_scale=14, color=BLUE, lw=2))
ax.text(1.55, 3.15, "Forward pass", fontsize=10, color=DARK, va="center")
ax.add_patch(FancyArrowPatch((0.5, 2.7), (1.4, 2.7), arrowstyle="-|>", mutation_scale=14,
                             color="#c0504d", lw=1.8, linestyle="--"))
ax.text(1.55, 2.7, "Safety override", fontsize=10, color=DARK, va="center")
ax.add_patch(FancyBboxPatch((0.5, 2.15), 0.55, 0.28, boxstyle="round,pad=0.01,rounding_size=0.06",
                            fc=HL, ec=BLUE, lw=1.6))
ax.text(1.15, 2.29, "Our contribution", fontsize=10, color=DARK, va="center")

# --- barra Experimental Protocol ---
ax.add_patch(FancyBboxPatch((3.4, 1.6), 12.2, 1.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="white", ec=BLUE, lw=1.4, linestyle="--"))
ax.text(3.75, 2.55, "Experimental\nProtocol", fontsize=12, fontweight="bold", color=BLUE, va="center")


def subbox(x, title, sub):
    ax.add_patch(FancyBboxPatch((x, 1.85), 2.7, 1.4, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=GRAYE, lw=1.1))
    ax.text(x + 1.35, 2.9, title, fontsize=10.5, fontweight="bold", color=DARK, ha="center")
    ax.text(x + 1.35, 2.25, sub, fontsize=8.8, color=DARK, ha="center", va="center")


subbox(5.7, "Dataset split", "train / test\n(no validation)")
subbox(8.7, "Baselines", "majority / keyword\nTF-IDF / SetFit")
subbox(11.7, "Metrics", "accuracy, macro-F1\nsafety-viol., latency")

fig.tight_layout()
fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches="tight")
print("salvato", OUT / "pipeline.png")

"""tester.py — tester interattivo di RUNMAXXIN.

Modelli DUE curve a punti di controllo — la FATICA (battito) e la VELOCITA' — e guardi come
la pipeline reagisce, canzone per canzone (6 secondi l'una). I BPM del target inseguono la
velocità che imposti; il battito guida sforzo e sicurezza. Sei tu il sensore.

Un "punto di controllo" = un numero; il tester interpola linearmente tra i punti per avere il
valore a ogni istante (vedi docs). HRR/sforzo/trend sono calcolati dal file base
`physiological_state.py`.

Uso:  python tester.py            # interattivo (chiede prompt, durata, profilo, curve)
"""
from __future__ import annotations

import argparse
import time

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from controller import decide
from symbolic import is_effort_compatible
from physiological_state import classify_effort, classify_trend, compute_hrr
from session import TOP_K, load_effort_by_song
import recommender

console = Console()
SPARK = "▁▂▃▄▅▆▇█"


def ask(label: str, default: str) -> str:
    try:
        raw = input(f"{label} [{default}]: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def nums(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split()]


def interp(points: list[float], frac: float) -> float:
    """Valore sulla curva a punti di controllo, alla frazione frac in [0,1] (interpolazione lineare)."""
    if len(points) == 1:
        return points[0]
    pos = frac * (len(points) - 1)
    i = int(pos)
    if i >= len(points) - 1:
        return points[-1]
    return points[i] + (points[i + 1] - points[i]) * (pos - i)


def spark_text(points: list[float], frac: float, n: int = 34) -> Text:
    """Sparkline ASCII della curva, col punto corrente evidenziato."""
    vals = [interp(points, k / (n - 1)) for k in range(n)]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    cur = int(round(frac * (n - 1)))
    t = Text()
    for k, v in enumerate(vals):
        ch = SPARK[min(7, int((v - lo) / rng * 7 + 0.5))]
        t.append(ch, style="bold black on cyan" if k == cur else "cyan")
    return t


def get_intent(prompt: str):
    try:
        from intent import route
        return route(prompt), True
    except Exception:
        from intent import GOAL_PARAMS, bpm_from_speed, goal_from_keywords, parse_numbers
        goal = goal_from_keywords(prompt) or "ModerateRun"
        numbers = parse_numbers(prompt)
        tb = bpm_from_speed(numbers["speed_kmh"]) if numbers.get("speed_kmh") else None
        return ({"goal": goal, "mood": "Neutral", "numbers": numbers,
                 "target_bpm": tb, "params": GOAL_PARAMS[goal]}, False)


def choose_song(target, top_df, effort_by_song):
    for i, (_, song) in enumerate(top_df.iterrows()):
        efforts = effort_by_song.get(str(song["song_id"]), "").split(";")
        if is_effort_compatible(efforts, target.effort_band):
            return song, i > 0
    return top_df.iloc[0], False


_REGIME_STYLE = {"recovery": "bold red", "warmup": "yellow",
                 "quantitative": "green", "qualitative": "cyan"}


def render(prompt, intent, nlp_real, fatigue_pts, speed_pts, frac,
           bpm_raw, speed, state, target, song, gate_hit):
    itxt = Text.assemble(
        ("▶ ", "bold"), (prompt, "italic white"),
        (f"    → goal={intent['goal']} mood={intent['mood']}", "white"),
        ("  (SetFit)" if nlp_real else "  (fallback keyword)", "dim"))
    header = Panel(itxt, title="RUNMAXXIN — tester", border_style="magenta")

    curve = Table.grid(padding=(0, 1))
    curve.add_row(Text("FATICA (bpm)", style="red"), spark_text(fatigue_pts, frac),
                  Text(f"{bpm_raw:.0f}", style="bold red"))
    curve.add_row(Text("VELOCITÀ    ", style="green"), spark_text(speed_pts, frac),
                  Text(f"{speed:.1f} km/h", style="bold green"))
    curve_panel = Panel(curve, title="le tue curve (▮ = adesso)", border_style="blue")

    sens = Table.grid(padding=(0, 1))
    hrr = state["mean_hrr"]
    barcol = "green" if hrr < 0.70 else "yellow" if hrr < 0.85 else "red"
    bar = Text("█" * int(hrr * 12) + "░" * (12 - int(hrr * 12)), style=barcol)
    sens.add_row("HRR", Text.assemble(bar, (f" {hrr:.2f}", "white")))
    sens.add_row("sforzo", state["effort_state"])
    sens.add_row("trend", state["trend_state"])
    sensori = Panel(sens, title="SENSORI (dal tuo physiological_state)", border_style="blue")

    tgt = Table.grid(padding=(0, 1))
    tgt.add_row("bpm", f"{target.bpm}")
    tgt.add_row("energy", f"{target.energy}")
    reg = "RECUPERO ⚠" if target.recovery else target.regime
    tgt.add_row("regime", Text(reg, style=_REGIME_STYLE.get(target.regime, "white")))
    bersaglio = Panel(tgt, title="TARGET (i bpm inseguono la velocità)", border_style="green")

    now = Table.grid(padding=(0, 1))
    now.add_row(Text(str(song["title"]), style="bold white"))
    now.add_row(Text.assemble((f"{song['genre']}", "cyan"), ("  ·  ", "dim"),
                              (f"{song['bpm']} bpm", "white"),
                              ("   [gate ✓ sforzo corretto]" if gate_hit else "", "yellow")))
    now.add_row(Text(f"▶ {song.get('spotify_url', '')}", style="dim green"))
    playing = Panel(now, title="♪ ORA IN RIPRODUZIONE", border_style="bright_magenta")

    return Group(header, curve_panel, Columns([sensori, bersaglio], expand=True), playing)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tester interattivo RUNMAXXIN.")
    ap.add_argument("--seconds-per-song", type=float, default=6.0)
    ap.add_argument("--step-min", type=float, default=3.0, help="minuti di corsa per canzone")
    a = ap.parse_args()

    console.print("[bold magenta]RUNMAXXIN — tester interattivo[/bold magenta]  (Invio = valore di default)\n")
    prompt = ask("Prompt", "oggi voglio spingere tantissimo")
    duration = float(ask("Durata allenamento (min)", "30"))
    resting = float(ask("Battito a riposo", "55"))
    maxhr = float(ask("Battito massimo", "190"))
    fatigue_pts = nums(ask("Curva FATICA — battiti ai punti di controllo", "120 150 175 185 175"))
    speed_pts = nums(ask("Curva VELOCITÀ — km/h ai punti di controllo", "10 13 16 12 9"))

    intent, nlp_real = get_intent(prompt)
    effort_by_song = load_effort_by_song()
    n_steps = max(1, round(duration / a.step_min))

    console.print("\n[dim]▶ play…[/dim]")
    played, prev_bpm = [], None
    with Live(console=console, refresh_per_second=8, screen=False) as live:
        for s in range(n_steps):
            frac = s / (n_steps - 1) if n_steps > 1 else 0.0
            t_min = frac * duration
            bpm_raw = interp(fatigue_pts, frac)
            speed = interp(speed_pts, frac)
            hrr = compute_hrr(bpm_raw, resting, maxhr)
            slope = 0.0 if prev_bpm is None else (bpm_raw - prev_bpm) / (a.step_min * 60)
            state = {"mean_hrr": hrr, "effort_state": classify_effort(hrr),
                     "trend_state": classify_trend(slope), "mean_speed_kmh": speed}
            target = decide(intent, state, prev_bpm, elapsed_min=t_min)
            top = recommender.recommend(target, top_k=TOP_K, exclude_song_ids=played)
            song, gate_hit = choose_song(target, top, effort_by_song)
            played.append(str(song["song_id"]))
            prev_bpm = bpm_raw
            live.update(render(prompt, intent, nlp_real, fatigue_pts, speed_pts, frac,
                               bpm_raw, speed, state, target, song, gate_hit))
            time.sleep(a.seconds_per_song)
    console.print("\n[bold green]Allenamento finito.[/bold green]")


if __name__ == "__main__":
    main()

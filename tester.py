"""tester.py — tester interattivo di RUNMAXXIN.

Modelli DUE curve a punti di controllo — lo SFORZO (in %, 0-100) e la VELOCITÀ (km/h) — su una
DISTANZA che scegli tu (km), e guardi come la pipeline reagisce, canzone per canzone (6s l'una).

Come si legge:
- lo SFORZO % → HRR (riserva cardiaca): via Karvonen inverso diventa un battito, lo sforzo lo
  classifica il file base `physiological_state.py`; guida sicurezza/recupero.
- la VELOCITÀ → guida i BPM che la musica insegue (entrainment continuo).
- la DISTANZA + la velocità → determinano il TEMPO (piano ⇒ ci metti di più ⇒ più canzoni).

Un "punto di controllo" = un numero; il tester interpola linearmente tra i punti.

Uso:  python tester.py            # interattivo
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
from physiological_state import classify_effort, classify_trend
from session import TOP_K, load_effort_by_song
import recommender

console = Console()
SPARK = "▁▂▃▄▅▆▇█"
N_SUB = 240          # sotto-passi per integrare la distanza
MAX_SONGS = 40       # tetto di sicurezza sul numero di canzoni


def ask(label: str, default: str) -> str:
    try:
        raw = input(f"{label} [{default}]: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def nums(s: str) -> list[float]:
    return [float(x) for x in s.replace(",", " ").split()]


def interp(points: list[float], frac: float) -> float:
    if len(points) == 1:
        return points[0]
    pos = frac * (len(points) - 1)
    i = int(pos)
    if i >= len(points) - 1:
        return points[-1]
    return points[i] + (points[i + 1] - points[i]) * (pos - i)


def spark_text(points: list[float], frac: float, n: int = 34) -> Text:
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


def plan_songs(distance_km, speed_pts, step_min):
    """Integra la velocità sulla distanza -> lista di step-canzone (frac, tempo_min).
    Piano => stesso km richiede più tempo => più canzoni in quel tratto."""
    steps, cum_time, next_song = [], 0.0, 0.0
    seg_km = distance_km / N_SUB
    for i in range(N_SUB):
        f = i / (N_SUB - 1)
        v = max(0.5, interp(speed_pts, f))          # km/h (evita divisioni per ~0)
        cum_time += seg_km / v * 60.0               # minuti per questo tratto
        if cum_time >= next_song and len(steps) < MAX_SONGS:
            steps.append((f, cum_time))
            next_song += step_min
    return steps, cum_time


_REGIME_STYLE = {"recovery": "bold red", "warmup": "yellow",
                 "quantitative": "green", "qualitative": "cyan"}


def render(prompt, intent, nlp_real, effort_pts, speed_pts, f, pct, disp_bpm,
           speed, t_min, state, target, song, gate_hit, top_df):
    itxt = Text.assemble(
        ("▶ ", "bold"), (prompt, "italic white"),
        (f"    → goal={intent['goal']} mood={intent['mood']}", "white"),
        ("  (SetFit)" if nlp_real else "  (fallback keyword)", "dim"),
        (f"    t={t_min:.1f} min", "dim white"))
    header = Panel(itxt, title="RUNMAXXIN — tester", border_style="magenta")

    curve = Table.grid(padding=(0, 1))
    curve.add_row(Text("SFORZO %", style="red"), spark_text(effort_pts, f),
                  Text(f"{pct:.0f}%  (~{disp_bpm:.0f} bpm)", style="bold red"))
    curve.add_row(Text("VELOCITÀ", style="green"), spark_text(speed_pts, f),
                  Text(f"{speed:.1f} km/h", style="bold green"))
    curve_panel = Panel(curve, title="le tue curve (▮ = adesso)", border_style="blue")

    sens = Table.grid(padding=(0, 1))
    hrr = state["mean_hrr"]
    bcol = "green" if hrr < 0.70 else "yellow" if hrr < 0.85 else "red"
    bar = Text("█" * int(hrr * 12) + "░" * (12 - int(hrr * 12)), style=bcol)
    sens.add_row("HRR", Text.assemble(bar, (f" {hrr:.2f}", "white")))
    sens.add_row("sforzo", state["effort_state"])
    sens.add_row("trend", state["trend_state"])
    sensori = Panel(sens, title="SENSORI (physiological_state)", border_style="blue")

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

    top3 = Table(expand=True, border_style="dim")
    for col in ("#", "canzone", "genere", "bpm", "P%", ""):
        top3.add_column(col)
    chosen_id = str(song["song_id"])
    for rank, (_, r) in enumerate(top_df.head(3).iterrows(), 1):
        mark = "◀ scelta" if str(r["song_id"]) == chosen_id else ""
        style = "bold white" if mark else "white"
        top3.add_row(str(rank), Text(str(r["title"])[:24], style=style), str(r["genre"]),
                     f"{r['bpm']:.0f}", f"{r['probability_percent']:.1f}", Text(mark, style="green"))
    candidati = Panel(top3, title="TOP 3 candidati (recommender)", border_style="dim")

    return Group(header, curve_panel, Columns([sensori, bersaglio], expand=True), playing, candidati)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tester interattivo RUNMAXXIN.")
    ap.add_argument("--seconds-per-song", type=float, default=6.0)
    ap.add_argument("--step-min", type=float, default=3.0, help="minuti di corsa per canzone")
    a = ap.parse_args()

    console.print("[bold magenta]RUNMAXXIN — tester interattivo[/bold magenta]  (Invio = default)\n")
    prompt = ask("Prompt", "oggi voglio spingere tantissimo")
    distance = float(ask("Distanza da percorrere (km)", "8"))
    resting = float(ask("Battito a riposo", "55"))
    maxhr = float(ask("Battito massimo", "190"))
    effort_pts = nums(ask("Curva SFORZO — % (0-100) ai punti di controllo", "40 60 80 92 80"))
    speed_pts = nums(ask("Curva VELOCITÀ — km/h ai punti di controllo", "10 13 16 12 8"))

    intent, nlp_real = get_intent(prompt)
    effort_by_song = load_effort_by_song()
    steps, total_time = plan_songs(distance, speed_pts, a.step_min)
    console.print(f"\n[dim]{distance:.1f} km · ~{total_time:.0f} min · {len(steps)} canzoni · ▶ play…[/dim]")

    played, prev_bpm = [], None
    with Live(console=console, refresh_per_second=8, screen=False) as live:
        for f, t_min in steps:
            pct = interp(effort_pts, f)
            hrr = max(0.0, min(1.2, pct / 100.0))               # % sforzo -> HRR
            disp_bpm = resting + hrr * (maxhr - resting)         # Karvonen inverso (per display/trend)
            speed = interp(speed_pts, f)
            slope = 0.0 if prev_bpm is None else (disp_bpm - prev_bpm) / (a.step_min * 60)
            state = {"mean_hrr": hrr, "effort_state": classify_effort(hrr),
                     "trend_state": classify_trend(slope), "mean_speed_kmh": speed}
            target = decide(intent, state, prev_bpm, elapsed_min=t_min)
            top = recommender.recommend(target, top_k=TOP_K, exclude_song_ids=played)
            song, gate_hit = choose_song(target, top, effort_by_song)
            played.append(str(song["song_id"]))
            prev_bpm = disp_bpm
            live.update(render(prompt, intent, nlp_real, effort_pts, speed_pts, f, pct,
                               disp_bpm, speed, t_min, state, target, song, gate_hit, top))
            time.sleep(a.seconds_per_song)
    console.print("\n[bold green]Allenamento finito.[/bold green]")


if __name__ == "__main__":
    main()

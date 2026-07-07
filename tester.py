"""tester.py — tester di RUNMAXXIN: PROMPT + scelta di un ARCHETIPO di prestazione -> play.

Niente più curve/distanza a mano (era un controsenso: la distanza è un risultato, non un input).
Scegli una frase e uno dei profili di corsa realistici generati da `simulate_sessions.py`, e
guardi come la pipeline reagisce, canzone per canzone (6s l'una): sensori, target che insegue
la velocità, canzone scelta e le Top-3 candidate. Stesso profilo con prompt diversi (o viceversa)
mostra come cambia la musica — è la matrice prompt × prestazione.

Uso:  python tester.py
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
from session import TOP_K, aggregate, group_by_song, load, load_effort_by_song
from simulate_sessions import ARCHETYPES
import recommender

console = Console()
SPARK = "▁▂▃▄▅▆▇█"
WINDOWS = "data/processed/physiological_windows.csv"

DESCRIPTIONS = {
    "steady": "ritmo costante in zona, nessun affaticamento",
    "push_fatigue": "parte forte poi cede: cuore su, velocità giù",
    "negative_split": "parte piano e accelera, ben gestito",
    "intervals": "ripetute: velocità e cuore che oscillano",
    "easy_recovery": "corsetta blanda, sforzo basso costante",
    "beginner_struggle": "erratico: cuore alto anche piano, pause camminata",
}


def ask(label: str, default: str) -> str:
    try:
        raw = input(f"{label} [{default}]: ").strip()
    except EOFError:
        raw = ""
    return raw or default


def spark_series(values: list[float], n: int = 40) -> Text:
    if not values:
        return Text("")
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    t = Text()
    for k, v in enumerate(values[-n:]):
        ch = SPARK[min(7, int((v - lo) / rng * 7 + 0.5))]
        t.append(ch, style="bold black on cyan" if k == len(values[-n:]) - 1 else "cyan")
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


def render(prompt, intent, nlp_real, sid, t_min, hrr_hist, spd_hist,
           state, target, song, gate_hit, top_df):
    itxt = Text.assemble(
        ("▶ ", "bold"), (prompt, "italic white"),
        (f"    → goal={intent['goal']} mood={intent['mood']}", "white"),
        ("  (SetFit)" if nlp_real else "  (fallback)", "dim"),
        (f"    prestazione: {sid}   t={t_min:.0f} min", "dim white"))
    header = Panel(itxt, title="RUNMAXXIN — tester", border_style="magenta")

    trend = Table.grid(padding=(0, 1))
    trend.add_row(Text("cuore   ", style="red"), spark_series(hrr_hist),
                  Text(f"HRR {state['mean_hrr']:.2f}", style="bold red"))
    trend.add_row(Text("velocità", style="green"), spark_series(spd_hist),
                  Text(f"{state.get('mean_speed_kmh') or 0:.1f} km/h", style="bold green"))
    andamento = Panel(trend, title="andamento (▮ = adesso)", border_style="blue")

    sens = Table.grid(padding=(0, 1))
    hrr = state["mean_hrr"]
    bcol = "green" if hrr < 0.70 else "yellow" if hrr < 0.85 else "red"
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
        top3.add_row(str(rank), Text(str(r["title"])[:24], style="bold white" if mark else "white"),
                     str(r["genre"]), f"{r['bpm']:.0f}", f"{r['probability_percent']:.1f}",
                     Text(mark, style="green"))
    candidati = Panel(top3, title="TOP 3 candidati (recommender)", border_style="dim")

    return Group(header, andamento, Columns([sensori, bersaglio], expand=True), playing, candidati)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tester RUNMAXXIN (prompt × prestazione).")
    ap.add_argument("--seconds-per-song", type=float, default=6.0)
    ap.add_argument("--song-seconds", type=int, default=180, help="secondi di corsa per canzone")
    a = ap.parse_args()

    console.print("[bold magenta]RUNMAXXIN — tester[/bold magenta]  (Invio = default)\n")
    prompt = ask("Prompt", "oggi voglio spingere tantissimo")

    console.print("\n[bold]Scegli una prestazione:[/bold]")
    for i, (sid, _u, _r, _m, goal, dur, _fn) in enumerate(ARCHETYPES, 1):
        console.print(f"  {i}. [cyan]{sid}[/cyan] ({dur // 60} min) — {DESCRIPTIONS.get(sid, '')}")
    idx = int(ask("Numero prestazione", "2")) - 1
    sid = ARCHETYPES[idx][0]

    intent, nlp_real = get_intent(prompt)
    effort_by_song = load_effort_by_song()
    windows = [w for w in load(WINDOWS) if w["session_id"] == sid]
    blocks = group_by_song(windows, a.song_seconds)

    console.print(f"\n[dim]{sid} · {len(blocks)} canzoni · ▶ play…[/dim]")
    played, last_bpm, hrr_hist, spd_hist = [], None, [], []
    with Live(console=console, refresh_per_second=8, screen=False) as live:
        for block in blocks:
            state = aggregate(block)
            t_min = int(block[0]["window_start_second"]) / 60.0
            hrr_hist.append(state["mean_hrr"])
            spd_hist.append(state.get("mean_speed_kmh") or 0.0)
            target = decide(intent, state, last_bpm, elapsed_min=t_min)
            top = recommender.recommend(target, top_k=TOP_K, exclude_song_ids=played)
            song, gate_hit = choose_song(target, top, effort_by_song)
            played.append(str(song["song_id"]))
            last_bpm = float(song["bpm"])
            live.update(render(prompt, intent, nlp_real, sid, t_min, hrr_hist, spd_hist,
                               state, target, song, gate_hit, top))
            time.sleep(a.seconds_per_song)
    console.print("\n[bold green]Allenamento finito.[/bold green]")


if __name__ == "__main__":
    main()

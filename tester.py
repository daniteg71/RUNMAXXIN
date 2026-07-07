"""tester.py — RUNMAXXIN tester: a PROMPT + a chosen PERFORMANCE archetype -> playback.

No hand-drawn curves/distance (that was a contradiction: distance is an output, not an input).
Pick a sentence and one of the realistic run profiles produced by `simulate_sessions.py`, and
watch how the pipeline reacts, song by song (6s each): sensors, target (BPM follows speed), the
chosen song and the Top-3 candidates. Same profile with different prompts (or vice versa) shows
how the music changes — the prompt x performance matrix. At the end it prints a workout summary
with the session stats and the full playlist (every song, with BPM and genre).

Usage:  python tester.py
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from statistics import mean

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
    "steady": "steady pace in the target zone, no fatigue",
    "push_fatigue": "starts hard then fades: HR up, speed down",
    "negative_split": "starts easy and accelerates, well-paced",
    "intervals": "intervals: speed and HR oscillate",
    "easy_recovery": "easy recovery run, low steady effort",
    "beginner_struggle": "erratic: high HR even when slow, walk breaks",
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
    tail = values[-n:]
    lo, hi = min(tail), max(tail)
    rng = (hi - lo) or 1.0
    t = Text()
    for k, v in enumerate(tail):
        ch = SPARK[min(7, int((v - lo) / rng * 7 + 0.5))]
        t.append(ch, style="bold black on cyan" if k == len(tail) - 1 else "cyan")
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
        (f"    performance: {sid}   t={t_min:.0f} min", "dim white"))
    header = Panel(itxt, title="RUNMAXXIN — tester", border_style="magenta")

    trend = Table.grid(padding=(0, 1))
    trend.add_row(Text("heart", style="red"), spark_series(hrr_hist),
                  Text(f"HRR {state['mean_hrr']:.2f}", style="bold red"))
    trend.add_row(Text("speed", style="green"), spark_series(spd_hist),
                  Text(f"{state.get('mean_speed_kmh') or 0:.1f} km/h", style="bold green"))
    trend_panel = Panel(trend, title="trend so far (▮ = now)", border_style="blue")

    sens = Table.grid(padding=(0, 1))
    sens.add_row("effort", state["effort_state"])
    sens.add_row("trend", state["trend_state"])
    sensors = Panel(sens, title="SENSORS (physiological_state)", border_style="blue")

    tgt = Table.grid(padding=(0, 1))
    tgt.add_row("bpm", f"{target.bpm}")
    tgt.add_row("energy", f"{target.energy}")
    reg = "RECOVERY ⚠" if target.recovery else target.regime
    tgt.add_row("regime", Text(reg, style=_REGIME_STYLE.get(target.regime, "white")))
    target_panel = Panel(tgt, title="TARGET (BPM follows speed)", border_style="green")

    now = Table.grid(padding=(0, 1))
    now.add_row(Text(str(song["title"]), style="bold white"))
    now.add_row(Text.assemble((f"{song['genre']}", "cyan"), ("  ·  ", "dim"),
                              (f"{song['bpm']} bpm", "white"),
                              ("   [gate ✓ effort corrected]" if gate_hit else "", "yellow")))
    now.add_row(Text(f"▶ {song.get('spotify_url', '')}", style="dim green"))
    playing = Panel(now, title="♪ NOW PLAYING", border_style="bright_magenta")

    top3 = Table(expand=True, border_style="dim")
    for col in ("#", "song", "genre", "bpm", "P%", ""):
        top3.add_column(col)
    chosen_id = str(song["song_id"])
    for rank, (_, r) in enumerate(top_df.head(3).iterrows(), 1):
        mark = "◀ chosen" if str(r["song_id"]) == chosen_id else ""
        top3.add_row(str(rank), Text(str(r["title"])[:24], style="bold white" if mark else "white"),
                     str(r["genre"]), f"{r['bpm']:.0f}", f"{r['probability_percent']:.1f}",
                     Text(mark, style="green"))
    candidates = Panel(top3, title="TOP 3 candidates (recommender)", border_style="dim")

    return Group(header, trend_panel, Columns([sensors, target_panel], expand=True), playing, candidates)


def print_summary(sid, prompt, intent, playlist, hrr_hist, spd_hist):
    avg_hrr, peak_hrr = mean(hrr_hist), max(hrr_hist)
    avg_spd, peak_spd = mean(spd_hist), max(spd_hist)
    efforts = Counter(p["effort"] for p in playlist)
    recov = sum(1 for p in playlist if p["regime"] == "recovery")
    duration = playlist[-1]["t"] if playlist else 0

    stats = Table.grid(padding=(0, 2))
    stats.add_row(Text("Performance", style="bold"), sid)
    stats.add_row(Text("Prompt", style="bold"), prompt)
    stats.add_row(Text("Intent", style="bold"), f"{intent['goal']} · {intent['mood']}")
    stats.add_row(Text("Duration", style="bold"), f"~{duration:.0f} min")
    stats.add_row(Text("Songs played", style="bold"), str(len(playlist)))
    stats.add_row(Text("Avg / peak HRR", style="bold"), f"{avg_hrr:.2f} / {peak_hrr:.2f}")
    stats.add_row(Text("Avg / peak speed", style="bold"), f"{avg_spd:.1f} / {peak_spd:.1f} km/h")
    stats.add_row(Text("Effort mix", style="bold"), ", ".join(f"{k}: {v}" for k, v in efforts.most_common()))
    stats.add_row(Text("Recovery songs", style="bold"), str(recov))
    console.print(Panel(stats, title="WORKOUT SUMMARY", border_style="green"))

    pl = Table(title="PLAYLIST — every song chosen", expand=True, border_style="bright_magenta")
    for c in ("#", "time", "title", "artist", "genre", "bpm", "regime"):
        pl.add_column(c)
    for i, p in enumerate(playlist, 1):
        pl.add_row(str(i), f"{p['t']:.0f}m", str(p["title"])[:26], str(p["artist"])[:18],
                   p["genre"], f"{p['bpm']:.0f}", p["regime"])
    console.print(pl)


def main() -> None:
    ap = argparse.ArgumentParser(description="RUNMAXXIN tester (prompt x performance).")
    ap.add_argument("--seconds-per-song", type=float, default=6.0)
    ap.add_argument("--song-seconds", type=int, default=180, help="seconds of running per song")
    a = ap.parse_args()

    console.print("[bold magenta]RUNMAXXIN — tester[/bold magenta]  (Enter = default)\n")
    prompt = ask("Prompt", "today I want to push really hard")

    console.print("\n[bold]Choose a performance:[/bold]")
    for i, (sid, _u, _r, _m, goal, dur, _fn) in enumerate(ARCHETYPES, 1):
        console.print(f"  {i}. [cyan]{sid}[/cyan] ({dur // 60} min) — {DESCRIPTIONS.get(sid, '')}")
    idx = int(ask("Performance number", "2")) - 1
    sid = ARCHETYPES[idx][0]

    intent, nlp_real = get_intent(prompt)
    effort_by_song = load_effort_by_song()
    windows = [w for w in load(WINDOWS) if w["session_id"] == sid]
    blocks = group_by_song(windows, a.song_seconds)

    console.print(f"\n[dim]{sid} · {len(blocks)} songs · ▶ play…[/dim]")
    played, last_bpm, hrr_hist, spd_hist, playlist = [], None, [], [], []
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
            playlist.append({"t": t_min, "title": song["title"], "artist": song["artist"],
                             "genre": song["genre"], "bpm": float(song["bpm"]),
                             "regime": "recovery" if target.recovery else target.regime,
                             "effort": state["effort_state"]})
            live.update(render(prompt, intent, nlp_real, sid, t_min, hrr_hist, spd_hist,
                               state, target, song, gate_hit, top))
            time.sleep(a.seconds_per_song)

    console.print("\n[bold green]Workout finished.[/bold green]\n")
    if playlist:
        print_summary(sid, prompt, intent, playlist, hrr_hist, spd_hist)


if __name__ == "__main__":
    main()

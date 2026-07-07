"""tester.py — RUNMAXXIN tester: PROMPT + a PERFORMANCE archetype -> a freshly GENERATED session.

The test = (what you SAY) x (how the body BEHAVES) x chance:
- PROMPT  -> intent (goal/mood) + scale: if it states a distance ("20 km") or a duration
             ("40 min") the run lasts that long; if it states no number, the length is random.
- ARCHETYPE -> the *shape* of the run (steady / push-then-fade / intervals / ...), a behavior
             template, not fixed values.
- CHANCE  -> the actual heart-rate/speed values are generated with random noise around the
             shape, so every run is different (use --seed to reproduce one).

The generated BPM/speed go through the base `physiological_state.py` (HRR -> effort/trend), then
the controller and the recommender. No pre-baked CSV, no build_dataset here — the session is
built in memory. At the end it prints a workout summary and the full playlist.

Usage:  python tester.py            # interactive; --seed N to reproduce a run
"""
from __future__ import annotations

import argparse
import random
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
from physiological_state import classify_effort, classify_trend, compute_hrr
from session import (TOP_K, load_effort_by_song, load_song_variants, pick_song)
from simulate_sessions import ARCHETYPES
import recommender

console = Console()
SPARK = "▁▂▃▄▅▆▇█"

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
        from intent import GOAL_PARAMS, goal_from_keywords, parse_numbers
        goal = goal_from_keywords(prompt) or "ModerateRun"
        return ({"goal": goal, "mood": "Neutral", "numbers": parse_numbers(prompt),
                 "target_bpm": None, "params": GOAL_PARAMS[goal]}, False)


def session_length_sec(numbers: dict, fn, rng) -> tuple[int, str]:
    """Quanto dura la sessione: dai numeri del prompt, o random. Ritorna (secondi, etichetta)."""
    if numbers.get("duration_min"):
        return int(numbers["duration_min"] * 60), f"duration {numbers['duration_min']} min"
    if numbers.get("distance_km"):
        avg_spd = mean(fn(int(k / 40 * 1000), 1000)[1] for k in range(1, 41))   # forma normalizzata
        return int(numbers["distance_km"] / max(1.0, avg_spd) * 3600), f"distance {numbers['distance_km']} km"
    mins = rng.randint(18, 42)
    return mins * 60, f"random {mins} min"


def generate_session(fn, resting, maxhr, total_sec, song_sec, rng) -> list[dict]:
    """Genera i dati sensore per canzone: forma dall'archetipo `fn` + rumore stocastico; lo
    sforzo/HRR/trend li calcola il file base physiological_state.py."""
    n = max(1, round(total_sec / song_sec))
    states, prev_bpm = [], None
    for i in range(n):
        center = i * song_sec + song_sec / 2
        bpm, speed, _phase = fn(int(center), total_sec)
        bpm = max(60.0, bpm + rng.gauss(0, 2.0))
        speed = max(1.0, speed + rng.gauss(0, 0.35))
        hrr = compute_hrr(bpm, resting, maxhr)
        slope = 0.0 if prev_bpm is None else (bpm - prev_bpm) / song_sec
        prev_bpm = bpm
        states.append({"t_min": i * song_sec / 60.0, "mean_hrr": hrr,
                       "effort_state": classify_effort(hrr), "trend_state": classify_trend(slope),
                       "mean_speed_kmh": speed})
    return states


_REGIME_STYLE = {"recovery": "bold red", "warmup": "yellow",
                 "quantitative": "green", "qualitative": "cyan"}


def render(prompt, intent, nlp_real, sid, scale, t_min, hrr_hist, spd_hist,
           state, target, song, explored, top_rows):
    itxt = Text.assemble(
        ("▶ ", "bold"), (prompt, "italic white"),
        (f"    → goal={intent['goal']} mood={intent['mood']}", "white"),
        ("  (SetFit)" if nlp_real else "  (fallback)", "dim"),
        (f"    {sid} · {scale} · t={t_min:.0f} min", "dim white"))
    header = Panel(itxt, title="RUNMAXXIN — tester", border_style="magenta")

    tr = Table.grid(padding=(0, 1))
    tr.add_row(Text("heart", style="red"), spark_series(hrr_hist),
               Text(f"HRR {state['mean_hrr']:.2f}", style="bold red"))
    tr.add_row(Text("speed", style="green"), spark_series(spd_hist),
               Text(f"{state.get('mean_speed_kmh') or 0:.1f} km/h", style="bold green"))
    trend_panel = Panel(tr, title="trend so far (▮ = now)", border_style="blue")

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
                              ("   [explore]" if explored else "", "yellow")))
    now.add_row(Text(f"▶ {song.get('spotify_url', '')}", style="dim green"))
    playing = Panel(now, title="♪ NOW PLAYING", border_style="bright_magenta")

    top3 = Table(expand=True, border_style="dim")
    for col in ("#", "song", "genre", "bpm", "P%", ""):
        top3.add_column(col)
    chosen_id = str(song["song_id"])
    for rank, r in enumerate(top_rows[:3], 1):
        mark = "◀ chosen" if str(r["song_id"]) == chosen_id else ""
        top3.add_row(str(rank), Text(str(r["title"])[:24], style="bold white" if mark else "white"),
                     str(r["genre"]), f"{r['bpm']:.0f}", f"{r['probability_percent']:.1f}",
                     Text(mark, style="green"))
    candidates = Panel(top3, title="TOP 3 candidates (recommender)", border_style="dim")

    return Group(header, trend_panel, Columns([sensors, target_panel], expand=True), playing, candidates)


def print_summary(sid, scale, prompt, intent, playlist, hrr_hist, spd_hist):
    efforts = Counter(p["effort"] for p in playlist)
    stats = Table.grid(padding=(0, 2))
    for k, v in (("Performance", f"{sid}  ({scale})"), ("Prompt", prompt),
                 ("Intent", f"{intent['goal']} · {intent['mood']}"),
                 ("Duration", f"~{playlist[-1]['t']:.0f} min"), ("Songs played", str(len(playlist))),
                 ("Avg / peak HRR", f"{mean(hrr_hist):.2f} / {max(hrr_hist):.2f}"),
                 ("Avg / peak speed", f"{mean(spd_hist):.1f} / {max(spd_hist):.1f} km/h"),
                 ("Effort mix", ", ".join(f"{e}: {c}" for e, c in efforts.most_common())),
                 ("Recovery songs", str(sum(1 for p in playlist if p["regime"] == "recovery")))):
        stats.add_row(Text(k, style="bold"), str(v))
    console.print(Panel(stats, title="WORKOUT SUMMARY", border_style="green"))

    pl = Table(title="PLAYLIST — every song chosen", expand=True, border_style="bright_magenta")
    for c in ("#", "time", "title", "artist", "genre", "bpm", "regime"):
        pl.add_column(c)
    for i, p in enumerate(playlist, 1):
        pl.add_row(str(i), f"{p['t']:.0f}m", str(p["title"])[:26], str(p["artist"])[:18],
                   p["genre"], f"{p['bpm']:.0f}", p["regime"])
    console.print(pl)


def main() -> None:
    ap = argparse.ArgumentParser(description="RUNMAXXIN tester (prompt x performance, generated).")
    ap.add_argument("--seconds-per-song", type=float, default=6.0)
    ap.add_argument("--song-seconds", type=int, default=180, help="seconds of running per song")
    ap.add_argument("--seed", type=int, default=None, help="reproduce a run (default: random)")
    a = ap.parse_args()

    console.print("[bold magenta]RUNMAXXIN — tester[/bold magenta]  (Enter = default)\n")
    prompt = ask("Prompt", "today I want to push hard for 20 km")

    console.print("\n[bold]Choose a performance shape:[/bold]")
    for i, (sid, *_rest) in enumerate(ARCHETYPES, 1):
        console.print(f"  {i}. [cyan]{sid}[/cyan] — {DESCRIPTIONS.get(sid, '')}")
    idx = int(ask("Performance number", "2")) - 1
    sid, _uid, resting, maxhr, _goal, _dur, fn = ARCHETYPES[idx]

    intent, nlp_real = get_intent(prompt)
    intent = {**intent, "target_bpm": None}          # nel tester il passo lo dà la performance, non il prompt
    rng = random.Random(a.seed)                       # None = casuale ogni volta; --seed = riproducibile
    total_sec, scale = session_length_sec(intent["numbers"], fn, rng)
    states = generate_session(fn, resting, maxhr, total_sec, a.song_seconds, rng)

    effort_by_song = load_effort_by_song()
    variants = load_song_variants()
    console.print(f"\n[dim]{sid} · {scale} · {len(states)} songs · ▶ play…[/dim]")
    played, last_bpm, hrr_hist, spd_hist, playlist = [], None, [], [], []
    with Live(console=console, refresh_per_second=8, screen=False) as live:
        for state in states:
            hrr_hist.append(state["mean_hrr"])
            spd_hist.append(state["mean_speed_kmh"])
            target = decide(intent, state, last_bpm, elapsed_min=state["t_min"])
            top = recommender.recommend(target, top_k=TOP_K * 8, exclude_song_ids=played)
            song, explored, top_rows = pick_song(rng, target, top, effort_by_song)
            played.extend(variants.get((song["title"], song["artist"]), [str(song["song_id"])]))
            last_bpm = float(song["bpm"])
            playlist.append({"t": state["t_min"], "title": song["title"], "artist": song["artist"],
                             "genre": song["genre"], "bpm": float(song["bpm"]),
                             "regime": "recovery" if target.recovery else target.regime,
                             "effort": state["effort_state"]})
            live.update(render(prompt, intent, nlp_real, sid, scale, state["t_min"], hrr_hist,
                               spd_hist, state, target, song, explored, top_rows))
            time.sleep(a.seconds_per_song)

    console.print("\n[bold green]Workout finished.[/bold green]\n")
    if playlist:
        print_summary(sid, scale, prompt, intent, playlist, hrr_hist, spd_hist)


if __name__ == "__main__":
    main()

"""run_demo.py — demo animata di RUNMAXXIN.

Parte da un prompt, poi "riproduce" una sessione di corsa: a schermo scorrono, canzone
dopo canzone, i dati dei sensori (battito, sforzo, trend) e la canzone scelta in quel
momento (genere, BPM, link Spotify). Serve a VEDERE il flusso, e a registrarlo per il video.

Il flusso dei sensori è un REPLAY (dalle finestre di build_dataset), non un dispositivo
live — dichiarato onestamente. La pipeline è la stessa dell'app reale.

Uso:
  python run_demo.py                                  # sessione affaticamento, prompt di default
  python run_demo.py "corsa da maratona a 15 km/h" --session marathon_ontarget
  python run_demo.py "voglio spingere forte" --speed 0.3   # più veloce
"""
from __future__ import annotations

import argparse
import random
import time
import zlib

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from controller import decide
from session import (TOP_K, aggregate, group_by_song, load, load_effort_by_song,
                     load_song_variants, pick_song)
import recommender

console = Console()


def get_intent(prompt: str):
    """route() reale se ci sono i modelli SetFit; altrimenti fallback a parole-chiave."""
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


def _bar(value: float, width: int = 12) -> Text:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    color = "green" if value < 0.70 else "yellow" if value < 0.85 else "red"
    return Text("█" * filled + "░" * (width - filled), style=color)


_REGIME_STYLE = {"recovery": "bold red", "warmup": "yellow",
                 "quantitative": "green", "qualitative": "cyan"}


def render(prompt, intent, nlp_real, elapsed, state, target, song, gate_hit, history):
    head = Text.assemble(("▶ PROMPT  ", "bold"), (prompt, "italic white"))
    itxt = Text.assemble(
        ("INTENT  ", "bold"),
        (f"goal={intent['goal']}  mood={intent['mood']}  target_bpm={intent['target_bpm']}", "white"),
        ("   (NLP: SetFit)" if nlp_real else "   (NLP: fallback keyword — modelli assenti)",
         "dim green" if nlp_real else "dim yellow"))
    header = Panel(Group(head, itxt), title="RUNMAXXIN", border_style="magenta")

    sens = Table.grid(padding=(0, 1))
    sens.add_row("tempo", f"{elapsed:.1f} min")
    sens.add_row("HRR", Text.assemble(_bar(state["mean_hrr"]), (f"  {state['mean_hrr']:.2f}", "white")))
    sens.add_row("sforzo", state["effort_state"])
    sens.add_row("trend", state["trend_state"])
    sensori = Panel(sens, title="SENSORI (replay 30s)", border_style="blue")

    tgt = Table.grid(padding=(0, 1))
    tgt.add_row("bpm", f"{target.bpm}")
    tgt.add_row("energy", f"{target.energy}")
    tgt.add_row("valence", f"{target.valence}")
    regime = "RECUPERO ⚠" if target.recovery else target.regime
    tgt.add_row("regime", Text(regime, style=_REGIME_STYLE.get(target.regime, "white")))
    bersaglio = Panel(tgt, title="TARGET (controller)", border_style="green")

    now = Table.grid(padding=(0, 1))
    now.add_row(Text(str(song["title"]), style="bold white"))
    now.add_row(Text(f"{song['artist']}", style="dim"))
    now.add_row(Text.assemble((f"{song['genre']}", "cyan"), ("  ·  ", "dim"),
                              (f"{song['bpm']} bpm", "white"),
                              ("   [explore]" if gate_hit else "", "yellow")))
    now.add_row(Text(f"▶ {song.get('spotify_url', '')}", style="dim green"))
    playing = Panel(now, title="♪ ORA IN RIPRODUZIONE", border_style="bright_magenta")

    hist = Table(expand=True, border_style="dim")
    hist.add_column("t"); hist.add_column("HRR"); hist.add_column("regime")
    hist.add_column("canzone"); hist.add_column("genere"); hist.add_column("bpm")
    for h in history[-8:]:
        hist.add_row(*h)

    return Group(header, Columns([sensori, bersaglio], expand=True), playing,
                 Panel(hist, title="storico", border_style="dim"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo animata RUNMAXXIN.")
    ap.add_argument("prompt", nargs="*", help="prompt utente (<=20 parole)")
    ap.add_argument("--windows", default="data/processed/physiological_windows.csv")
    ap.add_argument("--session", default="push_then_fatigue")
    ap.add_argument("--song-seconds", type=int, default=120)
    ap.add_argument("--speed", type=float, default=1.0, help="secondi di pausa per canzone")
    a = ap.parse_args()
    prompt = " ".join(a.prompt) or "oggi voglio spingere tantissimo"

    windows = [w for w in load(a.windows) if not a.session or w["session_id"] == a.session]
    effort_by_song = load_effort_by_song()
    variants = load_song_variants()
    intent, nlp_real = get_intent(prompt)

    from rich.live import Live
    rng = random.Random(zlib.crc32((a.session or prompt).encode()))
    played, last_bpm, history = [], None, []
    console.print("[dim]Avvio sessione…[/dim]")
    with Live(console=console, refresh_per_second=8, screen=False) as live:
        for block in group_by_song(windows, a.song_seconds):
            state = aggregate(block)
            elapsed = int(block[0]["window_start_second"]) / 60.0
            target = decide(intent, state, last_bpm, elapsed_min=elapsed)
            top = recommender.recommend(target, top_k=TOP_K * 8, exclude_song_ids=played)
            song, gate_hit, _rows = pick_song(rng, target, top, effort_by_song)
            played.extend(variants.get((song["title"], song["artist"]), [str(song["song_id"])]))
            last_bpm = float(song["bpm"])
            regime = "RECUPERO" if target.recovery else target.regime
            history.append((f"{elapsed:.0f}m", f"{state['mean_hrr']:.2f}", regime,
                            str(song["title"])[:22], str(song["genre"]), f"{song['bpm']:.0f}"))
            live.update(render(prompt, intent, nlp_real, elapsed, state, target, song, gate_hit, history))
            time.sleep(a.speed)
    console.print("\n[bold green]Sessione terminata.[/bold green]")


if __name__ == "__main__":
    main()

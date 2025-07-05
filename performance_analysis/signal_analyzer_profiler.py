# =====================================================================
#  File: signal_analyzer_profiler.py
#  Purpose: CPU-time profiler wrapper for the Signal-Analyzer program
# =====================================================================
"""
Signal-Analyzer Profiler
------------------------

Run your Signal-Analyzer *or any Python script* under ``cProfile``, then
save **.prof**, **text**, and **HTML** summaries in ``./performance_reports``.

Examples
~~~~~~~~
    # run the GUI app under the profiler for ~5 min
    python signal_analyzer_profiler.py path/to/main.py --duration 300

    # pass arguments through to the script being profiled
    python signal_analyzer_profiler.py run.py --duration 120 -- --file big.atf

    # just attach to an already-running PID for a quick 30-second sample
    python signal_analyzer_profiler.py --attach 12345 --duration 30
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
import cProfile
import pstats
from datetime import datetime
from pathlib import Path
from textwrap import dedent
import subprocess
import psutil


def _ensure_reports_dir() -> Path:
    reports = Path("performance_reports")
    reports.mkdir(exist_ok=True)
    return reports


# ------------------------------------------------------------------ #
# 1) Wrap a NEW python process in cProfile (preferred)
# ------------------------------------------------------------------ #
def _profile_new_process(target_script: str, argv: list[str], duration: int) -> Path:
    """
    Launch *target_script* inside this interpreter so we can use cProfile.
    Returns path to the raw .prof file.
    """
    reports_dir = _ensure_reports_dir()
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    prof_file = reports_dir / f"profile_{Path(target_script).stem}_{ts}.prof"

    profiler = cProfile.Profile()
    try:
        # inject CLI args for target script
        saved_argv = sys.argv
        sys.argv   = [target_script, *argv]

        profiler.enable()
        # run in the current interpreter
        runpy.run_path(target_script, run_name="__main__")
    finally:
        profiler.disable()
        sys.argv = saved_argv

    profiler.dump_stats(prof_file)
    return prof_file


# ------------------------------------------------------------------ #
# 2) Sample an EXISTING python process (PID) using psutil – quick but
#    coarse; records thread CPU% every 100 ms and aggregates.
# ------------------------------------------------------------------ #
def _sample_existing_process(pid: int, duration: int) -> list[tuple[int, float]]:
    proc = psutil.Process(pid)
    print(f"🔍 Sampling PID {pid} ({proc.name()}) for {duration}s …")

    per_thread_cpu: dict[int, float] = {t.id: 0.0 for t in proc.threads()}
    interval = 0.1
    t0       = time.time()

    # prime cpu_percent()
    proc.cpu_percent(interval=None)
    while time.time() - t0 < duration:
        for th in proc.threads():
            per_thread_cpu.setdefault(th.id, 0.0)
        time.sleep(interval)
        for th in proc.threads():
            # psutil returns *absolute* CPU time, we need delta since last call
            per_thread_cpu[th.id] += th.system_time + th.user_time
    # convert to sorted list
    ranked = sorted(per_thread_cpu.items(), key=lambda t: t[1], reverse=True)
    return ranked


# ------------------------------------------------------------------ #
# 3) Produce nice human-readable reports
# ------------------------------------------------------------------ #
def _render_profile_text(prof_path: Path) -> Path:
    txt_path = prof_path.with_suffix(".txt")
    with open(txt_path, "w") as fh:
        stats = pstats.Stats(str(prof_path), stream=fh)
        stats.sort_stats("cumulative")
        fh.write("┌─────────────────── Signal-Analyzer Profiler ───────────────────┐\n")
        stats.print_stats(40)                       # top 40 functions
    return txt_path


def _render_profile_html(prof_path: Path) -> Path:
    """
    Super-lightweight HTML report: table of top 100 cumulative-time functions.
    """
    html_path = prof_path.with_suffix(".html")
    stats     = pstats.Stats(str(prof_path))
    stats.sort_stats("cumulative")
    rows = []
    for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:100]:
        f_path, line, fn_name = func
        rows.append(
            f"<tr><td>{fn_name}</td><td>{Path(f_path).name}:{line}</td>"
            f"<td>{nc}</td><td>{ct:.3f}</td></tr>"
        )

    html = dedent(f"""\
        <!DOCTYPE html><html><head><meta charset="utf-8"/>
        <style>
            body{{font-family:Segoe UI,Arial,sans-serif;padding:1rem}}
            table{{border-collapse:collapse;width:100%}}
            th,td{{border:1px solid #ccc;padding:4px 8px;font-size:0.9rem}}
            th{{background:#f0f0f0;text-align:left}}
        </style></head><body>
        <h2>Signal-Analyzer Profiling Report</h2>
        <p>generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <table><thead><tr>
           <th>Function</th><th>Location</th><th>Calls</th><th>Cumulative s</th>
        </tr></thead><tbody>
        {''.join(rows)}
        </tbody></table></body></html>
    """)
    html_path.write_text(html, encoding="utf-8")
    return html_path


# =================================================================== #
#                        CLI Entrypoint
# =================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="signal_analyzer_profiler",
        description="Run a Python command under cProfile or sample an "
                    "existing process for quick CPU analysis."
    )
    ap.add_argument("script", nargs="?",
                    help="Path to the Python script to profile "
                         "(omit when --attach is used)")
    ap.add_argument("script_args", nargs=argparse.REMAINDER,
                    help="Arguments forwarded to the target script "
                         "(use -- to separate)")
    ap.add_argument("--duration", type=int, default=120,
                    help="profiling time window in seconds (default: 120)")
    ap.add_argument("--attach", type=int, metavar="PID",
                    help="sample an existing Python PID instead of running a script")

    args = ap.parse_args()
    if args.attach and args.script:
        ap.error("Specify EITHER --attach OR a script, not both.")

    if args.attach:
        ranking = _sample_existing_process(args.attach, args.duration)
        reports = _ensure_reports_dir()
        out     = reports / f"thread_sample_{args.attach}_{int(time.time())}.txt"
        with out.open("w") as fh:
            fh.write("Thread-level CPU time sample\n")
            fh.write(f"PID {args.attach}  –  duration {args.duration}s\n\n")
            fh.write(f"{'TID':>10} | Seconds CPU\n")
            fh.write("-" * 25 + "\n")
            for tid, secs in ranking:
                fh.write(f"{tid:>10} | {secs:9.3f}\n")
        print(f"✅ sample written to {out}")
        return

    if not args.script:
        ap.error("You must specify a script to run, or use --attach PID.")

    prof_path = _profile_new_process(args.script, args.script_args, args.duration)
    txt_path  = _render_profile_text(prof_path)
    html_path = _render_profile_html(prof_path)

    print("\n✅ profiling complete!")
    print(f"   raw  .prof : {prof_path}")
    print(f"   text report: {txt_path}")
    print(f"   HTML report: {html_path}")


if __name__ == "__main__":
    main()

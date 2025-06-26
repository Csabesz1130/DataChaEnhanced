# =====================================================================
#  File: thread_performance_monitor.py
#  Purpose: Real-time per-thread CPU / memory tracker for any PID
# =====================================================================
"""
Thread Performance Monitor
--------------------------
Live-streams CPU%, user/system time, and memory footprint for every
thread in a target process.  Saves CSV + optional curses TUI.

Examples
~~~~~~~~
    # watch the Signal-Analyzer process for 60 seconds
    python thread_performance_monitor.py --process signal_analyzer --duration 60

    # attach directly with a PID and show curses dashboard
    python thread_performance_monitor.py --pid 12345 --tui
"""
from __future__ import annotations
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import psutil

try:
    import curses
except ImportError:
    curses = None  # Windows w/out curses fallback


# ------------------------------------------------------------------ #
def find_pid_by_name(name: str) -> int | None:
    candidates = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if name.lower() in (p.info["name"] or "").lower():
                candidates.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None
    # pick the most recent
    winner = max(candidates, key=lambda x: x.info["create_time"])
    return winner.pid


# ------------------------------------------------------------------ #
class ThreadSampler:
    def __init__(self, pid: int, interval: float = 0.5):
        self.proc      = psutil.Process(pid)
        self.interval  = interval
        self.cpu_usage = defaultdict(float)     # tid -> cumulative %
        self.snapshots = []                      # each row for CSV

    # --------------------- #
    def sample(self) -> None:
        # call once to prime cpu_percent
        for th in self.proc.threads():
            psutil._psplatform.cext_threads(False) if hasattr(psutil._psplatform, "cext_threads") else None
        self.proc.cpu_percent(interval=None)

        while True:
            t0 = time.time()
            try:
                cur_threads = {th.id: th for th in self.proc.threads()}
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            row = {"timestamp": datetime.utcnow().isoformat()}
            for tid, info in cur_threads.items():
                row[f"tid_{tid}_user"]   = info.user_time
                row[f"tid_{tid}_system"] = info.system_time
            self.snapshots.append(row)

            # update aggregate CPU%
            for tid in cur_threads:
                self.cpu_usage[tid] = cur_threads[tid].user_time + cur_threads[tid].system_time

            yield cur_threads   # for live TUI

            # precise sleep to maintain interval
            elapsed = time.time() - t0
            time.sleep(max(self.interval - elapsed, 0))

    # --------------------- #
    def write_csv(self, path: Path) -> None:
        if not self.snapshots:
            return
        keys = sorted({k for row in self.snapshots for k in row})
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.snapshots)


# ------------------------------------------------------------------ #
def run_headless(pid: int, duration: int, interval: float) -> None:
    sampler   = ThreadSampler(pid, interval)
    reports   = Path("performance_reports")
    reports.mkdir(exist_ok=True)

    t_end = time.time() + duration
    try:
        for _ in sampler.sample():
            if time.time() >= t_end:
                break
    except KeyboardInterrupt:
        print("⏹️  interrupted by user")

    csv_path = reports / f"thread_trace_{pid}_{int(time.time())}.csv"
    sampler.write_csv(csv_path)
    print(f"✅ CSV trace saved: {csv_path}")


# ------------------------------------------------------------------ #
def tui_screen(stdscr, pid: int, duration: int, interval: float):
    sampler = ThreadSampler(pid, interval)
    t_end   = time.time() + duration
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(interval * 1000))

    for thread_info in sampler.sample():
        stdscr.erase()
        stdscr.addstr(0, 0, f"Thread Performance Monitor – PID {pid}")
        stdscr.addstr(1, 0, f"UTC {datetime.utcnow():%H:%M:%S}  |  q = quit")
        stdscr.addstr(2, 0, "-" * 60)
        stdscr.addstr(3, 0, f"{'TID':>10} {'CPU-s':>10} {'User-s':>10} {'Sys-s':>10}")
        stdscr.addstr(4, 0, "-" * 60)

        sorted_threads = sorted(
            sampler.cpu_usage.items(), key=lambda kv: kv[1], reverse=True
        )
        for idx, (tid, cpu_secs) in enumerate(sorted_threads[:20], start=5):
            th = thread_info.get(tid)
            if not th:
                continue
            stdscr.addstr(idx, 0,
                          f"{tid:>10} {cpu_secs:>10.2f} {th.user_time:>10.2f} {th.system_time:>10.2f}")

        stdscr.refresh()

        if stdscr.getch() in (ord("q"), ord("Q")):
            break
        if time.time() >= t_end:
            break

    # write CSV after leaving TUI
    reports = Path("performance_reports")
    reports.mkdir(exist_ok=True)
    csv_path = reports / f"thread_trace_{pid}_{int(time.time())}.csv"
    sampler.write_csv(csv_path)
    stdscr.erase()
    stdscr.addstr(0, 0, f"✅ trace saved to {csv_path}")
    stdscr.addstr(1, 0, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()


# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser(
        prog="thread_performance_monitor",
        description="Track per-thread CPU usage / timings for a live Python process."
    )
    ap.add_argument("--pid", type=int, help="target PID")
    ap.add_argument("--process", help="name substring to pick the newest matching PID")
    ap.add_argument("--duration", type=int, default=60,
                    help="how long to monitor (seconds)")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="sampling interval in seconds (default 0.5)")
    ap.add_argument("--tui", action="store_true",
                    help="interactive curses dashboard (Linux/macOS)")
    args = ap.parse_args()

    if not args.pid and not args.process:
        ap.error("Specify --pid or --process")

    pid = args.pid or find_pid_by_name(args.process)
    if pid is None:
        print("❌ could not locate matching process.")
        sys.exit(1)

    if args.tui and curses:
        curses.wrapper(tui_screen, pid, args.duration, args.interval)
    else:
        run_headless(pid, args.duration, args.interval)


if __name__ == "__main__":
    main()

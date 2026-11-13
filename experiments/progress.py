#!/usr/bin/env python3
"""Progress of Virtuoso runs, read from each run's heartbeat.txt.

usage: python3 progress.py exp_k20-sanity [--instr 10000000] [-w 30]
"""
import argparse, glob, os, re, time

PAT = re.compile(r"Periodic at (\d+)")

def fmt(s):
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"

def last_icount(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            tail = f.read().decode(errors="ignore")
    except OSError:
        return None
    m = PAT.findall(tail)
    return int(m[-1]) if m else 0

def show(suite, instr):
    now = time.time()
    print(f"{'run':40s} {'progress':>9s} {'instr':>12s} {'elapsed':>9s} {'ETA':>9s}")
    for d in sorted(glob.glob(os.path.join(suite, "results", "*"))):
        name = os.path.basename(d)
        if os.path.exists(os.path.join(d, "sim.stats")):
            print(f"{name:40s} {'done':>9s}")
            continue
        hb = os.path.join(d, "heartbeat.txt")
        if not os.path.exists(hb):
            print(f"{name:40s} {'waiting':>9s}")
            continue
        ic = last_icount(hb)
        start = os.path.getctime(hb)
        elapsed = now - start
        pct = 100.0 * ic / instr if instr else 0
        eta = elapsed * (instr - ic) / ic if ic else 0
        stale = now - os.path.getmtime(hb)
        warn = "  (no update for %s)" % fmt(stale) if stale > 600 else ""
        print(f"{name:40s} {pct:8.1f}% {ic:12,d} {fmt(elapsed):>9s} "
              f"{fmt(eta) if ic else '?':>9s}{warn}")

ap = argparse.ArgumentParser()
ap.add_argument("suite", help="suite dir, e.g. exp_k20-sanity")
ap.add_argument("--instr", type=int, default=None,
                help="instructions per run (default: read from job_list.csv / guess 10M)")
ap.add_argument("-w", "--watch", type=int, default=0, help="refresh every N seconds")
a = ap.parse_args()

instr = a.instr
if instr is None:
    instr = 10_000_000
    try:
        txt = open(os.path.join(a.suite, "jobfile.sh")).read()
        m = re.search(r"stop-by-icount:(\d+)", txt)
        if m:
            instr = int(m.group(1))
    except OSError:
        pass

while True:
    if a.watch:
        os.system("clear")
    show(a.suite, instr)
    if not a.watch:
        break
    time.sleep(a.watch)

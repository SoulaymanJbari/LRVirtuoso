#!/usr/bin/env python3
"""
run_local.py - Execute a Virtuoso jobfile.sh locally, without SLURM.

Reads the sbatch lines produced by create_experiments.py, extracts the
run-sniper command and the --output/--error paths, and runs the commands
N at a time on the local machine. slurm.out / slurm.err are written at the
same place as with SLURM, so get_experiments_status.py keeps working.

Usage:
    python3 run_local.py experiments/exp_x/jobfile.sh -j 48
    python3 run_local.py experiments/exp_x/jobfile.sh -j 48 --dry-run
    python3 run_local.py experiments/exp_x/jobfile.sh -j 48 --skip-done
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_jobfile(path):
    jobs = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or "sbatch" not in line:
                continue
            out = re.search(r"--output=(\S+)", line)
            err = re.search(r"--error=(\S+)", line)
            name = re.search(r"-J\s+(\S+)", line)
            first, last = line.find('"'), line.rfind('"')
            if not (out and err) or first == -1 or last <= first:
                print(f"[warn] line {ln}: cannot parse, skipped", file=sys.stderr)
                continue
            command = line[first + 1:last].strip()
            # Keep the wrapper used by sbatch (sets up the environment)
            wrapper = None
            for tok in shlex.split(line[:first]):
                if tok.endswith("native_wrapper.sh"):
                    wrapper = tok
            jobs.append({
                "name": name.group(1) if name else f"line{ln}",
                "out": out.group(1),
                "err": err.group(1),
                "cmd": command,
                "wrapper": wrapper,
            })
    return jobs


def is_done(job):
    d = os.path.dirname(job["out"])
    return any(os.path.exists(os.path.join(d, p))
               for p in ("sim.stats", os.path.join("simulation", "sim.stats")))


def run_job(job):
    os.makedirs(os.path.dirname(job["out"]), exist_ok=True)
    if job["wrapper"]:
        argv = ["bash", job["wrapper"], job["cmd"]]
    else:
        argv = ["bash", "-c", job["cmd"]]
    t0 = time.time()
    with open(job["out"], "w") as fo, open(job["err"], "w") as fe:
        rc = subprocess.call(argv, stdout=fo, stderr=fe)
    return job, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Run a Virtuoso jobfile locally")
    ap.add_argument("jobfile")
    ap.add_argument("-j", "--jobs", type=int, default=32, help="parallel runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip runs whose output dir already has sim.stats")
    args = ap.parse_args()

    jobs = parse_jobfile(args.jobfile)
    if args.skip_done:
        before = len(jobs)
        jobs = [j for j in jobs if not is_done(j)]
        print(f"[info] skipping {before - len(jobs)} completed run(s)")
    print(f"[info] {len(jobs)} run(s), {args.jobs} in parallel")

    if args.dry_run:
        for j in jobs:
            print(f"{j['name']}\n    {j['cmd']}")
        return

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for fut in as_completed(futures):
            job, rc, dt = fut.result()
            if rc == 0:
                done += 1
                tag = "OK  "
            else:
                failed += 1
                tag = f"FAIL({rc})"
            print(f"[{done + failed}/{len(jobs)}] {tag} {job['name']} ({dt/60:.1f} min)",
                  flush=True)
    print(f"[info] finished: {done} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

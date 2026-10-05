#!/usr/bin/env python3
"""Push small_track_colab.ipynb to Kaggle, wait for it to finish, and pull the results.

    .venv/bin/python scripts/kaggle_run.py [--timeout SECONDS] [--poll SECONDS] [--submit]

Each call bumps the Kaggle kernel to a new version, waits for it to finish, and
downloads everything it wrote into kaggle_runs/<timestamp>/ (config.json,
history.csv, the checkpoint, the submission CSV, experiment_log.csv, and the
executed notebook itself with every cell's output -- read that one when a run
errors out, it has the traceback).

Training never submits to Kaggle on its own (SUBMIT is forced False inside the
kernel, see small_track_colab.ipynb Section 12). Pass --submit to submit the
pulled CSV from here afterwards -- do that deliberately, not on every run; mind
the daily submission cap.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUSH_DIR = ROOT / "kaggle" / "small_track"
NOTEBOOK = ROOT / "small_track_colab.ipynb"
RUNS_DIR = ROOT / "kaggle_runs"
KERNEL = "mateogonzalezalonso/eurosat-small-track"
COMPETITION = "7-854-1-00-machine-learning-2026-coding-challenge"
KAGGLE = [sys.executable, "-m", "kaggle"]

DONE_STATUSES = {"COMPLETE", "ERROR", "CANCELLED"}


def run(args, **kw):
    print("+", " ".join(args))
    return subprocess.run(args, check=True, text=True, **kw)


def kernel_status():
    r = subprocess.run(KAGGLE + ["kernels", "status", KERNEL], capture_output=True, text=True)
    m = re.search(r'status "KernelWorkerStatus\.(\w+)"', r.stdout)
    if not m:
        raise RuntimeError(f"could not parse status: {r.stdout!r} {r.stderr!r}")
    return m.group(1)


def push():
    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(NOTEBOOK, PUSH_DIR / NOTEBOOK.name)
    run(KAGGLE + ["kernels", "push", "-p", str(PUSH_DIR)])


def wait(poll_seconds, timeout_seconds):
    start = time.monotonic()
    last = None
    while True:
        status = kernel_status()
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] status: {status}")
            last = status
        if status in DONE_STATUSES:
            return status
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"kernel still {status} after {timeout_seconds}s")
        time.sleep(poll_seconds)


def pull(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run(KAGGLE + ["kernels", "output", KERNEL, "-p", str(out_dir)])
    return out_dir


def summarize(out_dir):
    runs = sorted((out_dir / "runs").glob("*")) if (out_dir / "runs").exists() else []
    if not runs:
        print("no runs/<RUN_NAME>/ directory in the output -- check the executed notebook for errors")
        return None
    run_dir = runs[-1]
    print(f"\nrun: {run_dir.name}")
    cfg = run_dir / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        print(f"  params={c.get('N_PARAMS')} epochs={c.get('EPOCHS')} bands={len(c.get('USE_BANDS', []))} "
              f"norm={c.get('NORM_MODE')} smoke={c.get('SMOKE_TEST')}")
    log = out_dir / "experiment_log.csv"
    if log.exists():
        last_line = log.read_text().strip().splitlines()[-1]
        print(f"  experiment_log.csv (last row): {last_line}")
    sub = run_dir / "submission_small.csv"
    print(f"  submission: {sub if sub.exists() else 'NOT FOUND'}")
    return sub if sub.exists() else None


def submit(sub_path, message):
    run(KAGGLE + ["competitions", "submit", "-c", COMPETITION, "-f", str(sub_path), "-m", message])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", type=int, default=30, help="seconds between status checks (default 30)")
    ap.add_argument("--timeout", type=int, default=3600, help="give up after this many seconds (default 3600)")
    ap.add_argument("--submit", action="store_true", help="submit the resulting CSV to Kaggle afterwards")
    ap.add_argument("--message", default=None, help="submission message (default: auto from run name)")
    args = ap.parse_args()

    push()
    status = wait(args.poll, args.timeout)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / stamp
    pull(out_dir)
    print(f"\npulled to {out_dir}")

    if status != "COMPLETE":
        print(f"kernel finished with status {status} -- see {out_dir / NOTEBOOK.name} for the traceback")
        sys.exit(1)

    sub_path = summarize(out_dir)
    if args.submit:
        if not sub_path:
            print("nothing to submit")
            sys.exit(1)
        submit(sub_path, args.message or f"small track | {out_dir.name}")


if __name__ == "__main__":
    main()

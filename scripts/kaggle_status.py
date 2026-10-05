#!/usr/bin/env python3
"""Checks whether the Kaggle automation pipeline (see KAGGLE_WORKFLOW.md) is ready:
auth, competition access, the training-data cache (kernel + published dataset), and
whether the training kernel has been pushed before. Read-only, makes no changes.

Kaggle's API doesn't expose a progress percentage for a running kernel, only a
state (QUEUED/RUNNING/COMPLETE/ERROR) -- so for a kernel that's RUNNING, this
also reports how long it's been running, from "kernels list"'s lastRunTime. For
the actual progress lines (e.g. "decoded 15000/27000"), watch the kernel's own
page on kaggle.com -- its logs stream live there.

    .venv/bin/python scripts/kaggle_status.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone

KAGGLE = [sys.executable, "-m", "kaggle"]
COMPETITION = "7-854-1-00-machine-learning-2026-coding-challenge"
CACHE_KERNEL = "mateogonzalezalonso/eurosat-ms-cache-builder"
CACHE_DATASET = "mateogonzalezalonso/eurosat-ms-l1c-cache"
TRAIN_KERNEL = "mateogonzalezalonso/eurosat-small-track"


def sh(*args):
    return subprocess.run(KAGGLE + list(args), capture_output=True, text=True)


def ok(args):
    return sh(*args).returncode == 0


def kernel_state(slug):
    r = sh("kernels", "status", slug)
    if "KernelWorkerStatus." not in r.stdout:
        return None
    return r.stdout.split("KernelWorkerStatus.")[1].split('"')[0]


def kernel_elapsed(slug):
    # lastRunTime of the current version; only meaningful while QUEUED/RUNNING.
    r = sh("kernels", "list", "-m", "--format", "json")
    try:
        rows = json.loads(r.stdout)
    except ValueError:
        return None
    for row in rows:
        if row["ref"] == slug:
            started = datetime.fromisoformat(row["lastRunTime"]).replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) - started
    return None


def report_check(label, passed, ok_text, fail_text):
    print(f"{label:<28}" + (f"OK  {ok_text}" if passed else f"--  {fail_text}"))
    return passed


def report_kernel(label, slug, not_pushed_hint):
    state = kernel_state(slug)
    if state is None:
        print(f"{label:<28}{not_pushed_hint}")
    elif state in ("RUNNING", "QUEUED"):
        elapsed = kernel_elapsed(slug)
        suffix = f" ({int(elapsed.total_seconds() // 60)}m elapsed)" if elapsed else ""
        print(f"{label:<28}{state}{suffix}")
    else:
        print(f"{label:<28}{state}")


def main():
    print("Kaggle automation status")
    print("-------------------------")
    ready = True
    ready &= report_check("Kaggle auth:", ok(("config", "view")),
                           "token found", "no token - see KAGGLE_WORKFLOW.md (place one at ~/.kaggle/access_token)")
    ready &= report_check("Competition access:", ok(("competitions", "files", "-c", COMPETITION)),
                           "rules accepted", "join the competition / accept its rules on Kaggle")
    report_kernel("Cache-build kernel:", CACHE_KERNEL, "not pushed yet (make kaggle-cache-build)")
    ready &= report_check("Cache dataset:", ok(("datasets", "status", CACHE_DATASET)),
                           "published", "not published yet (make kaggle-cache)")
    report_kernel("Training kernel:", TRAIN_KERNEL, "not pushed yet (scripts/kaggle_run.py pushes it on first run)")
    print("-------------------------")
    if ready:
        print("READY: auth + competition access + cache dataset are in place.")
        print("Run: .venv/bin/python scripts/kaggle_run.py")
    else:
        print("NOT READY yet - fix the items marked -- above.")


if __name__ == "__main__":
    main()

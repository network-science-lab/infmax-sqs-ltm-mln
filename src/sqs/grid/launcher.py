"""Run grid jobs with bounded parallelism and report their progress."""

import os
import subprocess
import time
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

from sqs.grid.jobs import ROOT, Job, merge_slices, running_out_dirs

_REFRESH_SECONDS = 5.0


def available_cpus() -> int:
    """Return usable CPUs, honouring a cgroup v2 quota (``os.cpu_count`` ignores it)."""
    cpus = len(os.sched_getaffinity(0))
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if quota != "max":
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return cpus


def run(jobs: list[Job], workers: int, cuda_devices: list[str] | None = None) -> None:
    """Launch pending jobs as ``uv run sqs`` processes, at most ``workers`` at a time."""
    running_dirs = running_out_dirs()
    pending = [j for j in jobs if j.out_dir not in running_dirs and j.rows_done() < j.expected_rows]
    print(f"{len(jobs)} jobs: {len(jobs) - len(pending)} done or running, {len(pending)} to run")
    threads = max(1, available_cpus() // workers)
    running: dict[str, tuple[subprocess.Popen, TextIO]] = {}
    failed: list[str] = []
    bar = tqdm(total=len(pending), unit="job")
    launched = 0
    try:
        while pending or running:
            while pending and len(running) < workers:
                job = pending.pop(0)
                running[job.name] = _launch(job, threads, cuda_devices, launched)
                launched += 1
            for name, (proc, log) in list(running.items()):
                if proc.poll() is not None:
                    log.close()
                    del running[name]
                    bar.update()
                    if proc.returncode != 0:
                        failed.append(name)
                        tqdm.write(f"FAILED: {name} (see {log.name})")
            time.sleep(_REFRESH_SECONDS)
    finally:
        bar.close()
    print(f"Done: {bar.n} finished, {len(failed)} failed")
    merge_slices(jobs)


def _launch(
    job: Job, threads: int, cuda_devices: list[str] | None, index: int
) -> tuple[subprocess.Popen, TextIO]:
    """Start one job; its stdout / stderr go to ``<logs_dir>/<name>.log``."""
    job.logs_dir.mkdir(parents=True, exist_ok=True)
    log = (job.logs_dir / f"{job.name.replace('/', '__')}.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices[index % len(cuda_devices)]
    cmd = ["uv", "run", "sqs", f"--config-name={job.config_name}", *job.overrides]
    proc = subprocess.Popen(
        [*cmd, f"+run.max_threads={threads}"], cwd=ROOT, stdout=log, stderr=log, env=env
    )
    return proc, log


def print_status(jobs: list[Job], workers: int) -> None:
    """Print progress per group (``<prefix>/<source>``) and the estimated time left."""
    running_dirs = running_out_dirs()
    groups: dict[str, dict[str, float]] = {}
    longest = 0.0
    for job in jobs:
        rows = job.rows_done()
        if job.out_dir in running_dirs:
            state = "running"
        elif rows >= job.expected_rows:
            state = "done"
        else:
            state = "partial" if rows else "pending"
        group = groups.setdefault(
            job.name.split("@")[0].rsplit("/", 1)[0],
            {"done": 0, "running": 0, "partial": 0, "pending": 0, "hours": 0.0},
        )
        group[state] += 1
        hours = job.weight_seconds * max(0.0, 1 - rows / job.expected_rows) / 3600
        group["hours"] += hours
        longest = max(longest, hours)

    print(f"{'group':<22} {'done':>6} {'running':>8} {'partial':>8} {'pending':>8} {'~h left':>9}")
    for name, g in sorted(groups.items()):
        print(
            f"{name:<22} {g['done']:>6.0f} {g['running']:>8.0f} {g['partial']:>8.0f} "
            f"{g['pending']:>8.0f} {g['hours']:>9.1f}"
        )
    total = sum(g["hours"] for g in groups.values())
    wall = max(total / workers, longest)
    print(
        f"\n~{total:.0f} CPU-h left; with {workers} workers ~{wall:.0f} h (~{wall / 24:.1f} days), "
        f"longest job ~{longest:.0f} h"
    )

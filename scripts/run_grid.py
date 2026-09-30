#!/usr/bin/env python3
"""Run several pre-split ``sqs`` configs concurrently with bounded parallelism.

``configs/full/``, ``configs/arxiv/``, and ``configs/timik/`` each hold one config per
(protocol, network, mode, selector) combination (network fixed for the latter two folders), so
every file is already a small, self-contained job -- no runtime splitting needed.
``configs/full_experiment_aucs.yaml`` stays a single bundled config (aucs is cheap enough that
splitting it further would not help).

Jobs are launched as independent ``uv run sqs --config-name=<name>`` processes, sorted by
descending estimated weight (rough per-network timings) so the heaviest work starts first.
BLAS/OpenMP thread limits are applied in-process via ``run.max_threads`` (see ``spreading.run``),
not environment variables, so each job gets a fair share of cores instead of every process trying
to use them all.

Progress per job is read back from the tail of its ``main.log`` (no pipeline changes needed) and
shown as one ``tqdm`` bar per job.

Usage:
    uv run python scripts/run_grid.py "full/and_*" --workers 2 --cuda-devices 0
    uv run python scripts/run_grid.py "timik/*" "arxiv/*" --workers 2 --cuda-devices 0,1
"""

import csv
import fnmatch
import os
import re
import subprocess
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _ROOT / "configs"
_LAUNCHER_LOGS_DIR = _ROOT / "data" / "raw_results" / "_launcher_logs"

_RESULTS_FILE = "results.csv"
_CASE_RE = re.compile(r"Case: .*?:\s*\d+%\|.*?\|\s*(\d+)/(\d+)")
_REP_RE = re.compile(r"Rep:\s*\d+%\|.*?\|\s*(\d+)/(\d+)")
_TAIL_BYTES = 8192
_REFRESH_SECONDS = 5.0

# Rough per-case wall-clock cost in seconds
_NETWORK_SECONDS_PER_CASE = {
    "er2": 30.0,
    "er3": 70.0,
    "er5": 80.0,
    "sf2": 46.0,
    "sf3": 51.0,
    "sf5": 99.0,
    "aucs": 2.1,
    "ckm_physicians": 5.6,
    "lazega": 1.9,
    "arxiv_netscience_coauthorship": 3456.0,
    "timik1q2009": 5834.0,
}
_MODE_WEIGHT = {
    "SS": 1,
    "SQ": 2,
    "SQ_R": 3,
}
_DEFAULT_SECONDS_PER_CASE = 99.0  # unmeasured network: assume the heaviest known-light one


@dataclass
class Job:
    """One launchable config file."""

    name: str
    out_dir: str
    weight_seconds: float
    expected_rows: int


def resolve_config_names(patterns: list[str]) -> list[str]:
    """
    Expand config-name glob patterns against the YAML files in ``configs/`` (recursive).

    :param patterns: literal config names or ``fnmatch``-style globs, e.g. ``"full/and_*"``.

    :returns: sorted, de-duplicated list of matching config names (without the ``.yaml`` suffix,
        using ``/`` for subfolder configs, e.g. ``"full/and_er2_ss_mds_deg"``).
    """
    available = sorted(
        p.relative_to(_CONFIGS_DIR).with_suffix("").as_posix() for p in _CONFIGS_DIR.rglob("*.yaml")
    )
    matched: set[str] = set()
    for pattern in patterns:
        hits = [name for name in available if fnmatch.fnmatch(name, pattern)]
        if not hits:
            raise ValueError(f"Pattern {pattern!r} matched no config in {_CONFIGS_DIR}")
        matched.update(hits)
    return sorted(matched)


def build_job(config_name: str) -> Job:
    """
    Load one config and estimate its wall-clock weight.

    :param config_name: config name (without ``.yaml`` suffix, ``/`` for subfolders).

    :returns: the job for this config.
    """
    with (_CONFIGS_DIR / f"{config_name}.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    sp = cfg["spreading"]
    seeding_schedules_weight = sum(_MODE_WEIGHT[i['params']['mode']] for i in cfg["seeding_schedules"])
    cases = (
        len(sp["protocols"])
        * len(sp["mi_values"])
        * len(sp["seed_budgets"])
        * seeding_schedules_weight
    )
    network = cfg["networks"][0]
    weight = _NETWORK_SECONDS_PER_CASE.get(network, _DEFAULT_SECONDS_PER_CASE) * cases
    return Job(
        name=config_name,
        out_dir=str(cfg["run"]["out_dir"]),
        weight_seconds=weight,
        expected_rows=cases * sp["repetitions"],
    )


def already_running(config_name: str) -> bool:
    """
    Check whether a process for this config is already running, to avoid duplicate launches.

    :param config_name: config name (without ``.yaml`` suffix).

    :returns: True if a matching ``sqs --config-name=<config_name>`` process is alive.
    """
    # Launched jobs always have "+run.max_threads=<n>" trailing after "--config-name=<name>"
    # (see launch()), so anchoring on end-of-string here would never match; require a space
    # or end-of-string instead, so a name is not falsely matched as a prefix of a longer one.
    result = subprocess.run(
        ["pgrep", "-f", f"config-name={config_name}( |$)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def count_result_rows(out_dir: str) -> int:
    """
    Count data rows already written to a job's ``results.csv``.

    :param out_dir: job's configured output directory, relative to the repo root.

    :returns: number of completed result rows, or 0 if the file is missing or unreadable
        (e.g. truncated by a crash before a single full row was flushed).
    """
    results_path = _ROOT / out_dir / _RESULTS_FILE
    if not results_path.exists():
        return 0
    try:
        with results_path.open(newline="", encoding="utf-8") as fh:
            return max(0, sum(1 for _ in csv.reader(fh)) - 1)
    except (OSError, csv.Error):
        return 0


def read_progress(main_log: Path) -> float:
    """
    Estimate overall completion (0-100) for a running job from the tail of its ``main.log``.

    :param main_log: path to the job's Hydra log file.

    :returns: estimated completion percentage in [0, 100].
    """
    if not main_log.exists():
        return 0.0
    with main_log.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - _TAIL_BYTES))
        tail = fh.read().decode("utf-8", errors="ignore")

    case_matches = _CASE_RE.findall(tail)
    if not case_matches:
        return 0.0
    cases_done, cases_total = (int(v) for v in case_matches[-1])
    if cases_total == 0:
        return 0.0

    rep_matches = _REP_RE.findall(tail)
    rep_frac = 0.0
    if rep_matches:
        rep_done, rep_total = (int(v) for v in rep_matches[-1])
        rep_frac = rep_done / rep_total if rep_total else 0.0

    return 100.0 * (cases_done + rep_frac) / cases_total


def main() -> None:
    """Parse CLI arguments, resolve jobs, and run the bounded-concurrency batch."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs", nargs="+", help="Config names or fnmatch globs to run, e.g. 'full/and_*'"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Max concurrent jobs (default: 2 -- this machine is shared, not exclusive)",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids to round-robin via CUDA_VISIBLE_DEVICES "
        "(e.g. '0,1'). Configs hardcode 'cuda:0', so without this every job contends for it.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")
    cuda_devices = args.cuda_devices.split(",") if args.cuda_devices else None
    threads_per_job = max(1, (os.cpu_count() or args.workers) // args.workers)

    jobs = [build_job(name) for name in resolve_config_names(args.configs)]
    jobs.sort(key=lambda j: j.weight_seconds)
    # jobs.sort(key=lambda j: j.weight_seconds, reverse=True)

    print(
        f"Planned {len(jobs)} job(s) across {args.workers} worker(s), "
        f"{threads_per_job} threads/job:"
    )
    for j in jobs:
        print(f"  {j.name:<35} ~{j.weight_seconds / 3600:6.1f}h")

    skipped_running = [j for j in jobs if already_running(j.name)]
    for j in skipped_running:
        print(f"skipping {j.name}: already running")

    remaining = [j for j in jobs if j not in skipped_running]
    rows_done = {j.name: count_result_rows(j.out_dir) for j in remaining}
    skipped_complete = [j for j in remaining if rows_done[j.name] >= j.expected_rows]
    for j in skipped_complete:
        print(f"skipping {j.name}: already complete ({rows_done[j.name]}/{j.expected_rows} rows)")

    pending = [j for j in remaining if j not in skipped_complete]
    if not pending:
        print("Nothing to run.")
        return

    _LAUNCHER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    running: dict[str, subprocess.Popen] = {}
    log_handles: dict[str, TextIO] = {}
    bars: dict[str, tqdm] = {}
    out_dirs: dict[str, str] = {}
    finished: list[tuple[str, int]] = []

    def launch(job: Job, position: int) -> None:
        safe_name = job.name.replace("/", "__")
        log_fh = (_LAUNCHER_LOGS_DIR / f"{safe_name}.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        desc = job.name[:35].ljust(35)
        if cuda_devices:
            device = cuda_devices[position % len(cuda_devices)]
            env["CUDA_VISIBLE_DEVICES"] = device
            desc = f"[gpu {device}] {desc}"
        running[job.name] = subprocess.Popen(
            [
                "uv",
                "run",
                "sqs",
                f"--config-name={job.name}",
                f"+run.max_threads={threads_per_job}",
            ],
            cwd=_ROOT,
            stdout=log_fh,
            stderr=log_fh,
            env=env,
        )
        log_handles[job.name] = log_fh
        out_dirs[job.name] = job.out_dir
        bars[job.name] = tqdm(total=100, position=position, desc=desc, leave=True)

    try:
        while pending or running:
            while pending and len(running) < args.workers:
                launch(pending.pop(0), position=len(bars))

            for name in list(running):
                bars[name].n = round(read_progress(_ROOT / out_dirs[name] / "main.log"))
                bars[name].refresh()

                return_code = running[name].poll()
                if return_code is not None:
                    log_handles.pop(name).close()
                    bars[name].n = 100 if return_code == 0 else bars[name].n
                    bars[name].refresh()
                    bars[name].close()
                    del running[name]
                    finished.append((name, return_code))

            time.sleep(_REFRESH_SECONDS)
    finally:
        for bar in bars.values():
            if not bar.disable:
                bar.close()

    print()
    failed = [name for name, code in finished if code != 0]
    print(f"Done: {len(finished)} finished, {len(failed)} failed")
    for name in failed:
        safe_name = name.replace("/", "__")
        print(f"  FAILED: {name} (see {_LAUNCHER_LOGS_DIR / f'{safe_name}.log'})")


if __name__ == "__main__":
    main()

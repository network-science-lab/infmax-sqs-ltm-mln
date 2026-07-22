#!/usr/bin/env python3
"""Shard one or more base ``sqs`` configs across workers and run them with bounded concurrency.

Takes an existing config (e.g. ``full_experiment_and_ss``, which lists all 12 networks) and
splits its case grid into independently-launchable shards via Hydra CLI overrides -- no new YAML
files needed.  Networks whose estimated total cost would alone exceed a fair share of a worker's
time (``timik1q2009``, ``arxiv_netscience_coauthorship`` today) are further split by ``mi_value``
so they can run across several workers at once instead of being a single-network bottleneck; the
rest ("light" networks) are themselves balanced (longest-processing-time heuristic) into up to
``--workers`` groups instead of one monolithic bundle, so they also fill several workers at once.
Shards are sorted by descending estimated weight and fed into a worker pool of size ``--workers``,
so the heaviest work starts first and everything else backfills -- the closest this pipeline can
get to maximally saturating available GPUs/cores without changing how ``spreading.py`` itself
runs.

Each shard writes to its own subdirectory under the base config's ``out_dir`` (already-incremental
``results.csv`` per shard -- concatenate them for the full picture), and progress is read back from
the tail of each shard's ``main.log`` and rendered as one ``tqdm`` bar per shard.

Usage:
    uv run python scripts/run_grid.py full_experiment_and_ss --workers 8
    uv run python scripts/run_grid.py "full_experiment_and_*" --cuda-devices 0,1,2,3,4,5,6,7
"""

import argparse
import fnmatch
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _ROOT / "configs"

_CASE_RE = re.compile(r"Case: .*?:\s*\d+%\|.*?\|\s*(\d+)/(\d+)")
_REP_RE = re.compile(r"Rep:\s*\d+%\|.*?\|\s*(\d+)/(\d+)")

_TAIL_BYTES = 8192

# Rough per-case wall-clock cost in seconds, read off data/raw_results/*/main.log timings on
# 2026-07-13 (AND/OR, SS/SQ modes).  Only used to decide which networks need splitting and to
# order shards -- being off by a factor of ~2 does not change the outcome.
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
# Unmeasured network (e.g. l2_course_net_1): assume the heaviest *known-light* network rather
# than risk silently under-splitting an actually-heavy one.
_DEFAULT_SECONDS_PER_CASE = 99.0

_DEFAULT_WORKERS = 4


@dataclass
class Shard:
    """One independently-launchable slice of a base config's case grid."""

    label: str
    base_config: str
    networks: list[str]
    mi_values: list[float] | None
    out_dir: str
    weight_seconds: float


def resolve_config_names(patterns: list[str]) -> list[str]:
    """
    Expand config-name glob patterns against the YAML files in ``configs/``.

    :param patterns: literal config names or ``fnmatch``-style globs
        (e.g. ``"full_experiment_and_*"``).

    :returns: sorted, de-duplicated list of matching config names (without the ``.yaml`` suffix).
    """
    available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.yaml"))
    matched: set[str] = set()
    for pattern in patterns:
        hits = [name for name in available if fnmatch.fnmatch(name, pattern)]
        if not hits:
            raise ValueError(f"Pattern {pattern!r} matched no config in {_CONFIGS_DIR}")
        matched.update(hits)
    return sorted(matched)


def load_config(config_name: str) -> dict:
    """
    Load a base config's raw YAML.

    :param config_name: config name (without ``.yaml`` suffix).

    :returns: parsed config dict.
    """
    with (_CONFIGS_DIR / f"{config_name}.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _lpt_groups(items: list[str], weight: dict[str, float], n_groups: int) -> list[list[str]]:
    """
    Partition ``items`` into ``n_groups`` roughly-equal-weight buckets (longest-processing-time).

    Sorts items by descending weight and greedily drops each into the currently lightest bucket --
    the standard LPT heuristic, at most 4/3 off the optimal balance.

    :param items: items to partition (e.g. network names).
    :param weight: weight lookup for each item.
    :param n_groups: number of buckets to produce; capped at ``len(items)`` so none are empty.

    :returns: list of ``n_groups`` (or fewer) item buckets.
    """
    n_groups = max(1, min(n_groups, len(items)))
    buckets: list[list[str]] = [[] for _ in range(n_groups)]
    loads = [0.0] * n_groups
    for item in sorted(items, key=lambda i: weight[i], reverse=True):
        lightest = loads.index(min(loads))
        buckets[lightest].append(item)
        loads[lightest] += weight[item]
    return buckets


def build_shards(config_name: str, n_workers: int) -> list[Shard]:
    """
    Split a base config's networks into weighted shards sized for ``n_workers``.

    A network is classified "heavy" when its estimated total cost alone would exceed a fair
    ``1/n_workers`` share of the whole grid; such networks are split into one shard per
    ``mi_value`` so several workers can chip away at them concurrently.  The remaining ("light")
    networks are themselves LPT-balanced into up to ``n_workers`` groups -- rather than one
    monolithic bundle -- so light work also fills several workers concurrently instead of
    occupying just one for its whole (combined) duration.

    :param config_name: base config name (without ``.yaml`` suffix); must declare ``networks``,
        ``spreading.mi_values/seed_budgets``, and ``seeding_schedules``.
    :param n_workers: number of workers the shards will be scheduled across.

    :returns: shards sorted by descending estimated weight.
    """
    cfg = load_config(config_name)
    networks: list[str] = cfg["networks"]
    sp = cfg["spreading"]
    mi_values: list[float] = sp["mi_values"]
    cases_per_network = len(mi_values) * len(sp["seed_budgets"]) * len(cfg["seeding_schedules"])

    weight = {
        net: _NETWORK_SECONDS_PER_CASE.get(net, _DEFAULT_SECONDS_PER_CASE) * cases_per_network
        for net in networks
    }
    total_weight = sum(weight.values())
    fair_share = total_weight / n_workers if n_workers else total_weight

    base_out_dir = str(cfg["run"]["out_dir"]).rstrip("/")
    can_split_by_mi = len(mi_values) > 1
    heavy = [net for net in networks if weight[net] > fair_share and can_split_by_mi]
    light = [net for net in networks if net not in heavy]

    shards: list[Shard] = []
    for net in heavy:
        per_mi_weight = weight[net] / len(mi_values)
        for mi in mi_values:
            shards.append(
                Shard(
                    label=f"{config_name}/{net}/mi{mi}",
                    base_config=config_name,
                    networks=[net],
                    mi_values=[mi],
                    out_dir=f"{base_out_dir}/_shard_{net}_mi{mi}",
                    weight_seconds=per_mi_weight,
                )
            )

    for i, group in enumerate(_lpt_groups(light, weight, n_workers)):
        shards.append(
            Shard(
                label=f"{config_name}/light{i}({len(group)} nets)",
                base_config=config_name,
                networks=group,
                mi_values=None,
                out_dir=f"{base_out_dir}/_shard_light{i}",
                weight_seconds=sum(weight[net] for net in group),
            )
        )

    return sorted(shards, key=lambda s: s.weight_seconds, reverse=True)


def shard_overrides(shard: Shard) -> list[str]:
    """
    Build the Hydra CLI overrides that carve out one shard from its base config.

    :param shard: shard to launch.

    :returns: ``key=value`` override strings for ``uv run sqs --config-name=<base> <overrides>``.
    """
    overrides = [
        f"networks=[{','.join(shard.networks)}]",
        f"run.out_dir={shard.out_dir}",
    ]
    if shard.mi_values is not None:
        overrides.append(f"spreading.mi_values=[{','.join(str(v) for v in shard.mi_values)}]")
    return overrides


def already_running(out_dir: str) -> bool:
    """
    Check whether a process already targets this shard's ``out_dir``, to avoid duplicate launches.

    :param out_dir: the shard's ``run.out_dir`` override value.

    :returns: True if a matching process is alive.
    """
    result = subprocess.run(
        ["pgrep", "-f", f"run.out_dir={out_dir}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def read_progress(main_log: Path) -> float:
    """
    Estimate overall completion (0-100) for a running shard from the tail of its ``main.log``.

    Combines the last logged outer "Case: .../X/Y" line (cases fully completed so far) with the
    last logged inner "Rep: .../A/B" line (repetition progress within the current case) into a
    single smoothed percentage.  Best-effort: returns 0.0 before the first line is logged.

    :param main_log: path to the shard's Hydra log file.

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

    rep_matches = _REP_RE.findall(tail)
    rep_frac = 0.0
    if rep_matches:
        rep_done, rep_total = (int(v) for v in rep_matches[-1])
        rep_frac = rep_done / rep_total if rep_total else 0.0

    if cases_total == 0:
        return 0.0
    return 100.0 * (cases_done + rep_frac) / cases_total


def _detect_gpu_count() -> int:
    """
    Best-effort GPU count for the ``--workers`` default, via ``nvidia-smi -L``.

    :returns: number of visible GPUs, or ``_DEFAULT_WORKERS`` if detection fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False
        )
        count = len(result.stdout.strip().splitlines())
        return count or _DEFAULT_WORKERS
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _DEFAULT_WORKERS


def _detect_cpu_count() -> int:
    """
    Logical CPU count visible to this process (cgroup-aware where available).

    :returns: CPU count for ``--workers -1``, at least 1.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or _DEFAULT_WORKERS


def resolve_workers(value: int | None) -> int:
    """
    Resolve the ``--workers`` CLI value into a concrete worker count.

    ``None`` (unset) auto-detects the GPU count, falling back to ``_DEFAULT_WORKERS``.  ``-1``
    means "as many as the machine allows" -- all logical CPUs -- following the ``n_jobs=-1``
    convention from joblib/scikit-learn.  This can exceed the GPU count, which is fine for
    CPU-only shards but means GPU-heavy shards (``ts-net``/``voterank`` selectors) will contend
    for whatever ``--cuda-devices`` assigns them, or all pile onto ``cuda:0`` if left unset.

    :param value: raw ``--workers`` value.

    :returns: resolved worker count, at least 1.
    """
    if value is None:
        return _detect_gpu_count()
    if value == -1:
        return _detect_cpu_count()
    if value < 1:
        raise ValueError(f"--workers must be a positive int or -1, got {value}")
    return value


def main() -> None:
    """Parse CLI arguments, build shards, and run the bounded-concurrency batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs", nargs="+", help="Base config names or fnmatch globs to shard and run"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Max concurrent shards (default: detected GPU count, else 4). "
            "-1 means 'all logical CPUs' (joblib-style n_jobs=-1) -- can exceed the GPU count."
        ),
    )
    parser.add_argument(
        "--refresh", type=float, default=5.0, help="Progress refresh interval in seconds"
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help=(
            "Comma-separated physical GPU ids to round-robin across concurrent shards via "
            "CUDA_VISIBLE_DEVICES (e.g. '0,1,2,3'). Configs hardcode 'cuda:0', so without this "
            "every concurrent shard would contend for the same physical GPU."
        ),
    )
    args = parser.parse_args()
    cuda_devices = args.cuda_devices.split(",") if args.cuda_devices else None
    n_workers = resolve_workers(args.workers)

    gpu_count = _detect_gpu_count()
    if n_workers > gpu_count and not cuda_devices:
        print(
            f"warning: --workers {n_workers} exceeds the {gpu_count} detected GPU(s) and "
            "--cuda-devices is not set -- GPU-heavy shards (ts-net/voterank) will all contend "
            "for cuda:0"
        )

    config_names = resolve_config_names(args.configs)
    shards = [s for name in config_names for s in build_shards(name, n_workers)]
    shards.sort(key=lambda s: s.weight_seconds, reverse=True)

    print(f"Planned {len(shards)} shard(s) across {n_workers} worker(s):")
    for s in shards:
        print(f"  {s.label:<45} ~{s.weight_seconds / 3600:6.1f}h  -> {s.out_dir}")

    skipped = [s for s in shards if already_running(s.out_dir)]
    for s in skipped:
        print(f"skipping {s.label}: {s.out_dir} is already running")
    pending = [s for s in shards if s not in skipped]

    if not pending:
        print("Nothing to run.")
        return

    launcher_logs_dir = _ROOT / "data" / "raw_results" / "_launcher_logs"
    launcher_logs_dir.mkdir(parents=True, exist_ok=True)

    running: dict[str, subprocess.Popen] = {}
    log_handles: dict[str, TextIO] = {}
    bars: dict[str, tqdm] = {}
    out_dirs: dict[str, str] = {}
    finished: list[tuple[str, int]] = []

    def launch(shard: Shard, position: int) -> None:
        safe_name = shard.label.replace("/", "__")
        log_fh = (launcher_logs_dir / f"{safe_name}.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        desc = shard.label[:45].ljust(45)
        if cuda_devices:
            device = cuda_devices[position % len(cuda_devices)]
            env["CUDA_VISIBLE_DEVICES"] = device
            desc = f"[gpu {device}] {desc}"
        running[shard.label] = subprocess.Popen(
            ["uv", "run", "sqs", f"--config-name={shard.base_config}", *shard_overrides(shard)],
            cwd=_ROOT,
            stdout=log_fh,
            stderr=log_fh,
            env=env,
        )
        log_handles[shard.label] = log_fh
        out_dirs[shard.label] = shard.out_dir
        bars[shard.label] = tqdm(total=100, position=position, desc=desc, leave=True)

    try:
        while pending or running:
            while pending and len(running) < n_workers:
                launch(pending.pop(0), position=len(bars))

            for label in list(running):
                proc = running[label]
                pct = read_progress(_ROOT / out_dirs[label] / "main.log")
                bars[label].n = round(pct)
                bars[label].refresh()

                return_code = proc.poll()
                if return_code is not None:
                    log_handles.pop(label).close()
                    bars[label].n = 100 if return_code == 0 else bars[label].n
                    bars[label].refresh()
                    bars[label].close()
                    del running[label]
                    finished.append((label, return_code))

            time.sleep(args.refresh)
    finally:
        for bar in bars.values():
            if not bar.disable:
                bar.close()

    print("------------------------------------------------")
    failed = [label for label, code in finished if code != 0]
    print(f"Done: {len(finished)} finished, {len(failed)} failed")
    for label in failed:
        safe_name = label.replace("/", "__")
        print(f"  FAILED: {label} (see {launcher_logs_dir / f'{safe_name}.log'})")


if __name__ == "__main__":
    main()

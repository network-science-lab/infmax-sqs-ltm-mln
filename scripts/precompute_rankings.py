#!/usr/bin/env python3
r"""
Compute the cached Static rankings of a grid before running it.

Jobs then only read ``data/rankings/`` instead of waiting on the lock while the first job of a
network computes a ranking (mds on timik takes hours). Randomised selectors (mds) get one ranking
per repetition. With ``--scores`` the rankings go to ``data/rankings/v2`` instead, each actor with
the score it got (for analysis; simulations keep reading v1).

Usage:
    uv run python scripts/precompute_rankings.py configs/grids/static.yaml "static/*" \
        --device cpu --workers 16
    uv run python scripts/precompute_rankings.py configs/grids/static.yaml "static/*" --scores
"""

import json
import multiprocessing
import random
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sqs.grid.jobs import Job
from sqs.grid.spec import select
from sqs.loaders.net_loader import load_network
from sqs.pipeline import spreading
from sqs.runners.selectors import SELECTOR_CLASSES

# Rough ranking cost; heavy first
_ORDER = {"timik1q2009": 0, "arxiv_netscience_coauthorship": 1}
VERSION = "v2"

def _one_thread() -> None:
    """Limit BLAS / torch to one thread per worker process (one CPU per worker)."""
    import threadpoolctl  # noqa: PLC0415
    import torch  # noqa: PLC0415

    threadpoolctl.threadpool_limits(limits=1)
    torch.set_num_threads(1)


@lru_cache(maxsize=1)
def _network(name: str) -> Any:
    """Load a network once per worker process."""
    return load_network(name)


def _compute(network: str, params: dict[str, Any], rep: int | None) -> str:
    """Compute (or find) one cached ranking and describe it."""
    start = time.time()
    spreading.shared_ranking(_network(network), network, params, rep)
    rep_info = "" if rep is None else f" rep {rep}"
    return f"{network} {params['selector']}{rep_info}: {time.time() - start:.0f} s"


def scored_path(network: str, params: dict[str, Any], rep: int | None) -> Path:
    """v2 file of a ranking with scores: same name as the v1 cache file."""
    v1 = spreading.ranking_path(network, params, rep)
    return v1.parents[2] / VERSION / v1.parent.name / v1.name


def compute_scored(network: str, params: dict[str, Any], rep: int | None) -> str:
    """Compute one ranking with scores into v2 (seeded as the v1 cache) and describe it."""
    start = time.time()
    random.seed(rep or 0)
    np.random.seed(rep or 0)
    selector = SELECTOR_CLASSES[params["selector"]](**(params.get("selector_params") or {}))
    net = _network(network)
    if hasattr(selector, "scored"):
        scored = selector.scored(net)
    else:
        scored = [(a, None) for a in selector.actorwise(net)]
    path = scored_path(network, params, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    entries = [{"actor": a.actor_id, "score": score} for a, score in scored]
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(path)
    rep_info = "" if rep is None else f" rep {rep}"
    return f"{network} {params['selector']}{rep_info} (scores): {time.time() - start:.0f} s"


def _tasks(jobs: list[Job]) -> list[tuple[str, dict[str, Any], int | None]]:
    """Return unique (network, selector params, rep) of the Static jobs, heavy first."""
    tasks = {}
    for job in jobs:
        if job.schedule["class"] != "StaticSeedingStrategy":
            continue
        params = {
            k: v for k, v in job.schedule["params"].items() if k in ("selector", "selector_params")
        }
        selector = params["selector"]
        if selector in spreading.PER_REP_SELECTORS:
            reps: list[int | None] = list(range(job.repetitions))
        elif selector in spreading.DETERMINISTIC_SELECTORS:
            reps = [None]
        else:
            continue
        for rep in reps:
            tasks[(job.network, repr(sorted(params.items())), rep)] = (job.network, params, rep)
    return sorted(tasks.values(), key=lambda t: (_ORDER.get(t[0], 2), t[0], t[1]["selector"]))


def main() -> None:
    """Parse CLI arguments and compute the missing rankings in parallel."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="grid spec, e.g. configs/grids/static.yaml")
    parser.add_argument("patterns", nargs="+", help="globs over job names, e.g. 'static/*'")
    parser.add_argument("--device", help="replace selector_params.device, e.g. 'cpu'")
    parser.add_argument("--workers", type=int, default=2, help="parallel processes")
    parser.add_argument("--scores", action="store_true", help="rankings with scores into v2")
    args = parser.parse_args()

    needed = _tasks(select(args.spec, args.patterns, args.device))
    path, compute = (
        (scored_path, compute_scored) if args.scores else (spreading.ranking_path, _compute)
    )
    tasks = [t for t in needed if not path(*t).exists()]
    print(f"{len(needed)} rankings: {len(needed) - len(tasks)} cached, {len(tasks)} to compute")
    # spawn, not fork: a forked worker cannot initialise CUDA once the parent has touched it
    context = multiprocessing.get_context("spawn")
    failed = 0
    with ProcessPoolExecutor(args.workers, context, initializer=_one_thread) as pool:
        futures = {pool.submit(compute, *task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                print(f"[{i}/{len(tasks)}] {future.result()}", flush=True)
            except Exception as error:  # report and keep computing the other rankings
                failed += 1
                network, params, rep = futures[future]
                print(f"[{i}/{len(tasks)}] FAILED {network} {params} rep={rep}: {error!r}")
    print(f"Done: {len(tasks) - failed} ok, {failed} failed")


if __name__ == "__main__":
    main()

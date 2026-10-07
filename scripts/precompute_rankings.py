#!/usr/bin/env python3
r"""
Compute the cached Static rankings of a grid before running it.

Jobs then only read ``data/rankings/`` instead of waiting on the lock while the first job of a
network computes a ranking (mds on timik takes hours). Randomised selectors (mds) get one ranking
per repetition.

Usage:
    uv run python scripts/precompute_rankings.py configs/grids/static.yaml "static/*" \
        --device cpu --workers 16
"""

import multiprocessing
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqs.grid.jobs import Job
from sqs.grid.spec import select
from sqs.loaders.net_loader import load_network
from sqs.pipeline import spreading

# Rough ranking cost; heavy first
_ORDER = {"timik1q2009": 0, "arxiv_netscience_coauthorship": 1}


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
    args = parser.parse_args()

    needed = _tasks(select(args.spec, args.patterns, args.device))
    tasks = [t for t in needed if not spreading.ranking_path(*t).exists()]
    print(f"{len(needed)} rankings: {len(needed) - len(tasks)} cached, {len(tasks)} to compute")
    # spawn, not fork: a forked worker cannot initialise CUDA once the parent has touched it
    context = multiprocessing.get_context("spawn")
    failed = 0
    with ProcessPoolExecutor(args.workers, context, initializer=_one_thread) as pool:
        futures = {pool.submit(_compute, *task): task for task in tasks}
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

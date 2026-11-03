"""Pipeline stage: run sequential seeding experiments and save results to CSV."""

import fcntl
import hashlib
import itertools
import json
import os
import random
import zipfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import threadpoolctl
from tqdm import tqdm

from sqs.loaders.net_loader import load_network
from sqs.misc import utils
from sqs.misc.resources import monitor_resources
from sqs.misc.result_handler import (
    SimulationFullResult,
    append_result,
    extract_simulation_result,
)
from sqs.models.propagation.mltm import SQSMLTModel
from sqs.runners.selectors import SELECTOR_CLASSES
from sqs.runners.simulator import SequentialSimulator
from sqs.runners.strategies import (
    AdaptiveSeedingStrategy,
    MockSeedingStrategy,
    StaticSeedingStrategy,
    SubnetSeedingStrategy,
)

logger = utils.logger

_STRATEGY_CLASSES = {
    "AdaptiveSeedingStrategy": AdaptiveSeedingStrategy,
    "MockSeedingStrategy": MockSeedingStrategy,
    "StaticSeedingStrategy": StaticSeedingStrategy,
    "SubnetSeedingStrategy": SubnetSeedingStrategy,
}

# Selectors whose ranking depends only on network topology and can therefore be cached across
# repetitions.  RandomSeedSelector is intentionally excluded so that each rep still draws an
# independent random ordering.
DETERMINISTIC_SELECTORS = {
    "deg-c",
    "deg-c-d",
    "nghb-s",
    "nghb-sd",
    "voterank",
    "ts-net",
}
# Randomised selectors (mds: random search with a timeout): one cached ranking per rep, computed
# with seed(rep), so SS / SQ / SQ_R share MDS number r and the reps sample different MDSs.
PER_REP_SELECTORS = {"mds", "mds+deg"}

_RESULTS_FILE = "results.csv"
# Rankings are computed once per (network, selector) and shared by every job, so SS / SQ / SQ_R
_RANKINGS_VERSION = 1
_RANKINGS_DIR = Path(__file__).resolve().parents[3] / "data" / "rankings" / f"v{_RANKINGS_VERSION}"
_LOADED_RANKINGS: dict[Path, list[Any]] = {}
# Scored rankings: rep 0 is the ranking, reps 1..n-1 its replicas with random tie-breaks
# (scripts/replicate_rankings.py); mds has one ranking per rep
_REPLICAS_DIR = _RANKINGS_DIR.parent / "v2"
_DETAILED_LOGS_DIR = "detailed_logs"

# Unique key of a result row, used for resuming
_KEY_COLUMNS = ("network", "protocol", "mi_value", "seed_budget", "ss_method", "rep")
_KEY_DECIMALS = 6
ResultKey = tuple[str, str, float, float, str, int]


@contextmanager
def _log_sink(
    out_dir: Path, save: bool, use_zip: bool
) -> Generator[Callable[[str, str], Any], None, None]:
    """
    Context manager that yields a ``write(name, data)`` callable.

    When ``save=False`` the callable is a no-op.  When ``use_zip=True`` logs are written into
    ``detailed_logs.zip`` inside ``out_dir``; otherwise each log is written as an individual
    file under ``detailed_logs/``.

    :param out_dir: directory where output artefacts are written.
    :param save: whether to persist detailed logs at all.
    :param use_zip: archive logs into a single zip instead of separate files.
    """
    if not save:
        yield lambda _name, _data: None
        return
    if use_zip:
        with zipfile.ZipFile(
            out_dir / "detailed_logs.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            yield zf.writestr
    else:
        logs_dir = out_dir / _DETAILED_LOGS_DIR
        logs_dir.mkdir(exist_ok=True)
        yield lambda name, data: (logs_dir / name).write_text(data, encoding="utf-8")


def _resolve_strategy(schedule_cfg: dict[str, Any]) -> Any:
    """
    Instantiate a ``SeedingStrategy`` from its config block.

    :param schedule_cfg: dict with keys ``"class"`` (strategy class name) and ``"params"``
        (forwarded as kwargs to the constructor).

    :returns: configured ``SeedingStrategy`` instance.
    """
    cls_name: str = schedule_cfg["class"]
    params: dict[str, Any] = schedule_cfg.get("params", {})
    cls = _STRATEGY_CLASSES.get(cls_name)
    if cls is None:
        raise ValueError(
            f"Unknown seeding_schedule class {cls_name!r}. Available: {list(_STRATEGY_CLASSES)}"
        )
    return cls(**params)


def ranking_path(net_name: str, ss_params: dict[str, Any], rep: int | None = None) -> Path:
    """Return the cache file of (network, selector[, rep]); the device is not part of the key."""
    params = {k: v for k, v in (ss_params.get("selector_params") or {}).items() if k != "device"}
    tag = ss_params["selector"]
    if params:
        tag += "__" + hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    if rep is not None:
        tag += f"__rep{rep}"
    return _RANKINGS_DIR / net_name / f"{tag}.json"


def replica_ranking(net: Any, net_name: str, ss_params: dict[str, Any], rep: int) -> list[Any]:
    """Load the v2 ranking of (network, selector) for ``rep``: the original or a replica."""
    tag = ranking_path(net_name, ss_params).stem
    path = _REPLICAS_DIR / net_name / f"{tag}__rep{rep}.json"
    if rep == 0 and not path.exists():
        path = path.with_name(f"{tag}.json")  # deterministic selector: rep 0 is the original
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/replicate_rankings.py")
    if path not in _LOADED_RANKINGS:
        entries = json.loads(path.read_text(encoding="utf-8"))
        _LOADED_RANKINGS[path] = [net.get_actor(e["actor"]) for e in entries]
    return _LOADED_RANKINGS[path]


def shared_ranking(
    net: Any, net_name: str, ss_params: dict[str, Any], rep: int | None = None
) -> list[Any]:
    """
    Load the cached ranking of (network, selector[, rep]), computing it once under a file lock.

    :param net: network to rank.
    :param net_name: network name (cache key).
    :param ss_params: schedule params with ``selector`` and optional ``selector_params``.
    :param rep: repetition for randomised selectors (also the RNG seed); ``None``: seed 0.

    :returns: actors of ``net`` in ranking order.
    """
    path = ranking_path(net_name, ss_params, rep)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not path.exists():  # another job may have computed it while we waited
                random.seed(rep or 0)
                np.random.seed(rep or 0)
                selector = SELECTOR_CLASSES[ss_params["selector"]](
                    **(ss_params.get("selector_params") or {})
                )
                ranking = [a.actor_id for a in selector.actorwise(net)]
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
                tmp.replace(path)
                logger.info("Saved ranking to %s", path)
            lock_path.unlink(missing_ok=True)
    if path not in _LOADED_RANKINGS:  # read each file once per process
        ids = json.loads(path.read_text(encoding="utf-8"))
        _LOADED_RANKINGS[path] = [net.get_actor(aid) for aid in ids]
    return _LOADED_RANKINGS[path]


def _result_key(
    network: str, protocol: str, mi_value: float, seed_budget: float, ss_method: str, rep: int
) -> ResultKey:
    """Build a result row key (floats rounded to match values read back from CSV)."""
    return (
        str(network),
        str(protocol),
        round(float(mi_value), _KEY_DECIMALS),
        round(float(seed_budget), _KEY_DECIMALS),
        str(ss_method),
        int(rep),
    )


def _repair_trailing_partial_row(path: Path) -> None:
    """Drop a last row cut off by a crash, so the next append does not glue onto it."""
    with path.open("rb+") as fh:
        data = fh.read()
        if not data or data.endswith(b"\n"):
            return
        last_newline = data.rfind(b"\n")
        fh.truncate(last_newline + 1 if last_newline >= 0 else 0)
    logger.warning("Dropped a partially written trailing row from %s", path)


def _load_completed_keys(path: Path) -> set[ResultKey]:
    """Return keys of rows already in a results CSV."""
    if not path.exists():
        return set()
    _repair_trailing_partial_row(path)
    if path.stat().st_size == 0:
        return set()
    # usecols keeps file column order
    df = pd.read_csv(path, usecols=list(_KEY_COLUMNS), encoding="utf-8")[list(_KEY_COLUMNS)]
    df = df.dropna()
    return {_result_key(*row) for row in df.itertuples(index=False, name=None)}


def run(config: dict[str, Any]) -> None:
    """
    Execute the spreading pipeline stage.

    Iterates over the cartesian product of networks, protocols, mi values,  seed budgets, seeding
    schedules, and repetitions; runs ``SequentialSimulator`` for each combination; appends each
    result to the CSV in ``out_dir`` as soon as it is computed. Reps already in that CSV are
    skipped unless ``run.resume`` is false.

    :param config: Hydra-composed pipeline configuration dict.
    """
    max_threads: int | None = config["run"].get("max_threads")
    if max_threads is not None:
        # Caps BLAS/OpenMP thread pools (numpy, scipy, sklearn) in-process rather than via
        # environment variables, so a launcher running several shards concurrently can give each
        # one a fair share of cores without oversubscribing the machine.
        threadpoolctl.threadpool_limits(limits=max_threads)
        import torch  # noqa: PLC0415

        torch.set_num_threads(max_threads)

    out_dir = Path(config["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = config["spreading"]
    repetitions: int = sp["repetitions"]
    max_epochs: int | None = sp["max_epochs"]
    patience: int | None = sp.get("patience")

    dl_cfg: dict[str, Any] = sp.get("detailed_logs", {})
    save_logs: bool = dl_cfg.get("save", False)
    zip_logs: bool = dl_cfg.get("zip", False)

    net_names: list[str] = config["networks"]
    protos: list[str] = sp["protocols"]
    mi_vals: list[float] = sp["mi_values"]
    seed_budgets: list[float] = sp["seed_budgets"]
    seeding_schedules: list[dict[str, Any]] = config["seeding_schedules"]
    ss_names = [s["name"] for s in seeding_schedules]
    if len(ss_names) != len(set(ss_names)):
        duplicates = {n for n in ss_names if ss_names.count(n) > 1}
        raise ValueError(f"Duplicate seeding_schedule names: {sorted(duplicates)}")

    cases = list(itertools.product(net_names, protos, mi_vals, seed_budgets, seeding_schedules))
    out_path = out_dir / _RESULTS_FILE
    results_written = 0
    reps_skipped = 0

    resume: bool = config["run"].get("resume", True)
    completed = _load_completed_keys(out_path) if resume else set()
    if completed:
        logger.info("Resuming: %d result rows already present in %s", len(completed), out_path)

    # A network depends only on its name, which does not vary with protocol, mi, or seed_budget.
    # Caching it here (instead of inside the per-case loop) avoids reparsing the same network
    # file on every one of the cartesian-product cases below.
    net_cache: dict[str, Any] = {}

    logger.info("Pipeline started")

    outer_pbar = tqdm(cases, desc="", colour="green", file=open(os.devnull, "w"), leave=False)
    with _log_sink(out_dir, save_logs, zip_logs) as write_log:
        for net_name, proto, mi_val, seed_budget, schedule_cfg in outer_pbar:
            outer_pbar.set_description(f"Case: {net_name}/{proto}/{mi_val}/{schedule_cfg['name']}")
            logger.info(str(outer_pbar))

            ss_name: str = schedule_cfg["name"]
            ss_class: str = schedule_cfg["class"]
            ss_params: dict[str, Any] = schedule_cfg.get("params", {})
            ss_selector: str = ss_params.get("selector", "")
            ss_mode: str = ss_params.get("mode", "")

            pending_reps = [
                rep
                for rep in range(repetitions)
                if _result_key(net_name, proto, mi_val, seed_budget, ss_name, rep) not in completed
            ]
            reps_skipped += repetitions - len(pending_reps)
            if not pending_reps:
                continue

            if net_name not in net_cache:
                net_cache[net_name] = load_network(net_name)
            net = net_cache[net_name]
            total_actors = net.get_actors_num()
            n_seeds = max(1, round(seed_budget * total_actors))

            # Static rankings come from the shared cache: one per (network, selector) for
            # deterministic selectors, one per rep for randomised ones (mds).
            # With ranking_replicas, rep r uses replica r instead (ties broken at random).
            static = schedule_cfg["class"] == "StaticSeedingStrategy"
            replicas = static and sp.get("ranking_replicas", False)
            precomputed_ranking = None
            if static and not replicas and ss_selector in DETERMINISTIC_SELECTORS:
                precomputed_ranking = shared_ranking(net, net_name, ss_params)

            inner_pbar = tqdm(pending_reps, leave=False, file=open(os.devnull, "w"))
            for rep in inner_pbar:
                logger.debug("Rep: %s", str(inner_pbar))

                # Seed both RNGs so that rep i of method A and rep i of method B share the same
                # random state — satisfying the paired-test assumption in compare_methods.py.
                random.seed(rep)
                np.random.seed(rep)

                # Fresh strategy, model and network copy per repetition so that state from previous
                # runs does not leak into the next one.
                strategy = _resolve_strategy(schedule_cfg)
                if replicas:
                    precomputed_ranking = replica_ranking(net, net_name, ss_params, rep)
                elif static and ss_selector in PER_REP_SELECTORS:
                    precomputed_ranking = shared_ranking(net, net_name, ss_params, rep)
                if precomputed_ranking is not None and isinstance(strategy, StaticSeedingStrategy):
                    strategy.set_precomputed_ranking(precomputed_ranking)

                model = SQSMLTModel(protocol=proto, mi_value=mi_val)
                sim = SequentialSimulator(model, net.copy())

                device = (ss_params.get("selector_params") or {}).get("device", "cpu")
                with monitor_resources(device) as resource_usage:
                    run_log = sim.perform_propagation(
                        n_seeds=n_seeds,
                        strategy=strategy,
                        max_epochs=max_epochs,
                        patience=patience,
                        track_full_logs=save_logs,
                    )

                partial = extract_simulation_result(run_log, total_actors, n_seeds)
                full_result = SimulationFullResult.from_partial(
                    partial=partial,
                    network=net_name,
                    protocol=proto,
                    seed_budget=seed_budget,
                    mi_value=mi_val,
                    ss_method=ss_name,
                    ss_class=ss_class,
                    ss_selector=ss_selector,
                    ss_mode=ss_mode,
                    rep=rep,
                    wall_seconds=resource_usage.wall_seconds,
                    cpu_seconds=resource_usage.cpu_seconds,
                    peak_rss_mb=resource_usage.peak_rss_mb,
                    peak_rss_delta_mb=resource_usage.peak_rss_delta_mb,
                    peak_gpu_mb=resource_usage.peak_gpu_mb,
                )
                append_result(full_result, out_path)
                results_written += 1

                # Filename encodes all varied dimensions so logs from different combinations never
                # collide inside the same zip / directory.
                log_name = (
                    f"{net_name}__{proto}__mi{mi_val}__b{seed_budget}__{ss_name}__r{rep:03d}.json"
                )
                write_log(log_name, json.dumps(run_log.logger.get_detailed_logs()))

                gpu_info = (
                    f"  gpu={resource_usage.peak_gpu_mb:.1f}MB"
                    if resource_usage.peak_gpu_mb is not None
                    else ""
                )
                logger.debug(
                    "rep=%d  net=%s  protocol=%s  mi=%.2f  budget=%.2f  ss=%s. gain=%.1f%%"
                    "  t=%.2fs  cpu_t=%.2fs  peak_rss=%.1fMB%s",
                    rep,
                    net_name,
                    proto,
                    mi_val,
                    seed_budget,
                    ss_name,
                    partial.gain,
                    resource_usage.wall_seconds,
                    resource_usage.cpu_seconds,
                    resource_usage.peak_rss_mb,
                    gpu_info,
                )

    if reps_skipped:
        logger.info("Skipped %d reps already present in %s", reps_skipped, out_path)
    logger.info("Saved %d results to %s", results_written, out_path)

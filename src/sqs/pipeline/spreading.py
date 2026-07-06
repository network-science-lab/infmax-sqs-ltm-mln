"""Pipeline stage: run sequential seeding experiments and save results to CSV."""

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
from tqdm import tqdm

from sqs.loaders.net_loader import load_network
from sqs.misc import utils
from sqs.misc.resources import monitor_resources
from sqs.misc.result_handler import (
    SimulationFullResult,
    extract_simulation_result,
    save_results,
)
from sqs.models.propagation.mltm import SQSMLTModel
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
_DETERMINISTIC_SELECTORS = {
    "deg-c",
    "deg-c-d",
    "nghb-s",
    "nghb-sd",
    "voterank",
    "mds",
    "mds+deg",
    "ts-net",
}

_RESULTS_FILE = "results.csv"
_DETAILED_LOGS_DIR = "detailed_logs"


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


def run(config: dict[str, Any]) -> None:
    """
    Execute the spreading pipeline stage.

    Iterates over the cartesian product of networks, protocols, mi values,  seed budgets, seeding
    schedules, and repetitions; runs ``SequentialSimulator`` for each combination; collects
    results; writes a single CSV to ``out_dir``.

    :param config: Hydra-composed pipeline configuration dict.
    """
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
    results: list[SimulationFullResult] = []

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

            net = load_network(net_name)
            total_actors = net.get_actors_num()
            n_seeds = max(1, round(seed_budget * total_actors))

            # Pre-compute the actor ranking for StaticSeedingStrategy with a
            # deterministic selector.  The ranking depends only on topology, so
            # it is identical across all repetitions and can be injected into
            # each rep's fresh strategy instance to avoid recomputing it 30×.
            precomputed_ranking = None
            if schedule_cfg["class"] == "StaticSeedingStrategy":
                selector_key = schedule_cfg.get("params", {}).get("selector", "")
                if selector_key in _DETERMINISTIC_SELECTORS:
                    _proto = _resolve_strategy(schedule_cfg)
                    assert isinstance(_proto, StaticSeedingStrategy)
                    precomputed_ranking = _proto.get_or_compute_ranking(net)

            inner_pbar = tqdm(range(repetitions), leave=False, file=open(os.devnull, "w"))
            for rep in inner_pbar:
                logger.debug("Rep: %s", str(inner_pbar))

                # Seed both RNGs so that rep i of method A and rep i of method B share the same
                # random state — satisfying the paired-test assumption in compare_methods.py.
                random.seed(rep)
                np.random.seed(rep)

                # Fresh strategy, model and network copy per repetition so that state from previous
                # runs does not leak into the next one.
                strategy = _resolve_strategy(schedule_cfg)
                if precomputed_ranking is not None and isinstance(strategy, StaticSeedingStrategy):
                    strategy.set_precomputed_ranking(precomputed_ranking)

                model = SQSMLTModel(protocol=proto, mi_value=mi_val)
                sim = SequentialSimulator(model, net.copy())

                with monitor_resources() as resource_usage:
                    run_log = sim.perform_propagation(
                        n_seeds=n_seeds,
                        strategy=strategy,
                        max_epochs=max_epochs,
                        patience=patience,
                    )

                partial = extract_simulation_result(run_log, total_actors, n_seeds)
                results.append(
                    SimulationFullResult.from_partial(
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
                        rss_delta_mb=resource_usage.rss_delta_mb,
                        peak_gpu_mb=resource_usage.peak_gpu_mb,
                    )
                )

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
                    "  t=%.2fs  cpu=%.1fMB%s",
                    rep,
                    net_name,
                    proto,
                    mi_val,
                    seed_budget,
                    ss_name,
                    partial.gain,
                    resource_usage.wall_seconds,
                    resource_usage.rss_delta_mb,
                    gpu_info,
                )

    out_path = out_dir / _RESULTS_FILE
    save_results(results, out_path)
    logger.info("Saved %d results to %s", len(results), out_path)

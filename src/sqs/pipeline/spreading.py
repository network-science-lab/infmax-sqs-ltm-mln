"""Pipeline stage: run sequential seeding experiments and save results to CSV."""

import itertools
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from sqs import utils
from sqs.loaders.net_loader import load_network
from sqs.models.mltm import SQSMLTModel
from sqs.result_handler import (
    SimulationFullResult,
    extract_simulation_result,
    save_results,
)
from sqs.runners.seeding_strategy import MockSeedingStrategy
from sqs.runners.simulator import SequentialSimulator

logger = utils.logger

_STRATEGY_CLASSES = {
    "MockSeedingStrategy": MockSeedingStrategy,
}

_RESULTS_FILE = "results.csv"
_DETAILED_LOGS_DIR = "detailed_logs"


def _resolve_strategy(schedule_cfg: dict[str, Any]) -> Any:
    """
    Instantiate a ``SeedingStrategy`` from its config block.

    :param schedule_cfg: dict with keys ``"class"`` (strategy class name) and
        ``"params"`` (forwarded as kwargs to the constructor).

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

    Iterates over the cartesian product of networks, protocols, mi values,
    seed budgets, and repetitions; runs ``SequentialSimulator`` for each
    combination; collects results; and writes a single CSV to ``out_dir``.

    :param config: Hydra-composed pipeline configuration dict.
    """
    out_dir = Path(config["run"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / _DETAILED_LOGS_DIR
    logs_dir.mkdir(exist_ok=True)

    sp = config["spreading"]
    schedule_cfg: dict[str, Any] = sp["seeding_schedule"]
    max_epochs: int = schedule_cfg["max_epochs"]
    patience: int | None = schedule_cfg.get("patience")
    repetitions: int = sp["repetitions"]
    ss_method: str = schedule_cfg["class"]

    network_names: list[str] = config["networks"]
    protocols: list[str] = sp["protocols"]
    mi_values: list[float] = sp["mi_values"]
    seed_budgets: list[float] = sp["seed_budgets"]

    combinations = list(itertools.product(network_names, protocols, mi_values, seed_budgets))
    results: list[SimulationFullResult] = []

    for net_name, protocol, mi_value, seed_budget in tqdm(
        combinations, desc="Experiments", colour="green"
    ):
        logger.info("Loading network %s", net_name)
        net = load_network(net_name)
        total_actors = net.get_actors_num()
        n_seeds = max(1, round(seed_budget * total_actors))

        for rep in tqdm(
            range(repetitions), desc=f"{net_name}/{protocol}/mi={mi_value}", leave=False
        ):
            strategy = _resolve_strategy(schedule_cfg)
            model = SQSMLTModel(protocol=protocol, mi_value=mi_value)
            sim = SequentialSimulator(model, net.copy())

            run_log = sim.perform_propagation(
                n_seeds=n_seeds,
                strategy=strategy,
                max_epochs=max_epochs,
                patience=patience,
            )

            partial = extract_simulation_result(run_log, total_actors)
            results.append(
                SimulationFullResult.from_partial(
                    partial=partial,
                    network=net_name,
                    protocol=protocol,
                    seed_budget=seed_budget,
                    mi_value=mi_value,
                    ss_method=ss_method,
                )
            )

            log_name = f"{net_name}__{protocol}__mi{mi_value}__b{seed_budget}__r{rep:03d}.json"
            log_path = logs_dir / log_name
            with log_path.open("w", encoding="utf-8") as fh:
                json.dump(run_log.logger.get_detailed_logs(), fh)

            logger.debug(
                "rep=%d  net=%s  protocol=%s  mi=%.2f  budget=%.2f  gain=%.1f%%",
                rep,
                net_name,
                protocol,
                mi_value,
                seed_budget,
                partial.gain,
            )

    out_path = out_dir / _RESULTS_FILE
    save_results(results, out_path)
    logger.info("Saved %d results to %s", len(results), out_path)
